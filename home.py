import streamlit as st
import graphviz as graphviz
import networkx as nx
import pandas as pd
import requests 
import autogen
from codegpt_chatgroups import CodeGPTAssistantAgent, CodeGPTStandaloneAssistantAgent
from codegpt_chatgroups.graph_chatgroup import GraphGroupChat
import hmac

# HACK: use st.write instead of print
import builtins
def st_write_print(*args, **kwargs):
    args = [arg.replace("\033[33m", "**").replace("\033[0m", "**") for arg in args]
    st.write((' '.join(map(str, args))))
builtins.print = st_write_print

title = "AutoGen with CodeGPT Agents"
st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")
st.title(title)
st.sidebar.header(title)

padding_left = 5
st.markdown(
        f"""<style>
        .appview-container .main .block-container{{
        padding-left: {padding_left}rem;}}
            .stMultiSelect [data-baseweb="tag"] {{
                height: fit-content;
            }}
            .stMultiSelect [data-baseweb="tag"] span[title] {{
                white-space: normal; max-width: 100%; overflow-wrap: anywhere;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False



if not check_password():
    st.stop()  # Do not continue if check_password is not True.

col1, col2 = st.columns([2,1])
with col1.expander("NOTES", expanded=True):
    st.write("""
    - *It requires an OpenAI API key for external agents*
    - *It assumes that the **name** of each CodeGPT agent describes what it does*
    - *It assumes that the **prompt** of each CodeGPT agent describes the knowledge contrains*
    - *It's recommended to not perform interactions between CodeGPT agents directly. Prefer to use an agent who controls the conversation.*
    - *`User_proxy` is the starting point*
    """)



if not "codegpt_api_key" in st.session_state:
    st.session_state["codegpt_api_key"] = None

if not "openai_api_key" in st.session_state:
    st.session_state["openai_api_key"] = None

if "agent_list" not in st.session_state:
    st.session_state["agent_list"] = None

CODEGPT_API_KEY = st.sidebar.text_input("CodeGPT API Key", type="password", value=st.session_state.get("codegpt_api_key"))
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key"))
st.session_state["openai_api_key"] = OPENAI_API_KEY

if not CODEGPT_API_KEY:
    st.sidebar.warning("Please enter your CodeGPT API key.", icon="⚠️")
    st.stop()
else:
    st.session_state["codegpt_api_key"] = CODEGPT_API_KEY



if not st.session_state["agent_list"]:
    authorization_header = {"Authorization": f"Bearer {CODEGPT_API_KEY}"}
    response = requests.get("https://api.codegpt.co/v1/agent", headers=authorization_header)
    agent_list = {}

    if response.status_code == 200:
        for agent in response.json():
            agent_list[agent["name"].replace(" ", "_")] = agent
        st.session_state["agent_list"] = agent_list
    else:
        st.sidebar.error(f"Incorrect API Key")
        st.stop()

if not st.session_state["agent_list"]:
    st.error("Error: No agents found")
    st.stop()

codegpt_system_messages = {k: v["prompt"] for k, v in st.session_state["agent_list"].items()}


other_agents_list = ["Planner", "PlanFollower"] #, "Critic"]

other_agents_system_messages = {
    "Planner": "System message for Planner",
    "PlanFollower": "System message for PlanFollower",
    #"Critic": "System message for Critic",
}


if "codegpt_edges" not in st.session_state:
    st.session_state["codegpt_edges"] = []

if "other_edges" not in st.session_state:
    st.session_state["other_edges"] = []

if "other_agents" not in st.session_state:
    st.session_state["other_agents"] = other_agents_system_messages

if "codegpt_agents_list" not in st.session_state:
    st.session_state["codegpt_agents_list"] = []

if "other_agents_list" not in st.session_state:
    st.session_state["other_agents_list"] = []

if "graph" not in st.session_state:
    st.session_state["graph"] = None


codegpt_agents = st.sidebar.multiselect("CodeGPT agents", list(codegpt_system_messages.keys()), default=st.session_state["codegpt_agents_list"])
other_agents = st.sidebar.multiselect("Other agents", list(st.session_state["other_agents"].keys()), default=st.session_state["other_agents_list"])
other_agents.append("User_proxy")

codegpt_possible_edges = []
other_possible_edges = []
for agent in codegpt_agents:
    for other_agent in codegpt_agents + other_agents:
        if other_agent != agent:
            codegpt_possible_edges.append(f"{agent} ↔️ {other_agent}")

for other_agent in other_agents:
    for _other_agent in other_agents:
        if _other_agent != other_agent:
            other_possible_edges.append(f"{other_agent} → {_other_agent}")


with st.sidebar.expander("Add another agent"):
    with st.form("new_agent"):
        name = st.text_input("Name")
        system_message = st.text_area("System message")
        new_agent = st.form_submit_button("Add agent")
if new_agent and name and system_message:
    if name in st.session_state["other_agents"]:
        st.sidebar.error("Agent already exists")
        st.stop()
    st.session_state["other_agents"].update({name: system_message})
    st.rerun()

_col1, col3 = st.columns([2, 1])

with _col1.form("new_graph"):
    col1, col2 = st.columns(2)
    codegpt_edges = col1.multiselect("CodeGPT Edges", options=codegpt_possible_edges, key="codegpt_edges",
                                    default=[item for item in st.session_state["codegpt_edges"] if item in codegpt_possible_edges])

    other_edges = col2.multiselect("Other edges", options=other_possible_edges, key="other_edges",
                                default=[item for item in st.session_state["other_edges"] if item in other_possible_edges])
    edges = codegpt_edges + other_edges
    new_graph = st.form_submit_button("Update graph")


if edges and new_graph:
    G = nx.DiGraph()
    nodes = set()
    for edge in edges:
        if "↔️" in edge:
            nodes.add(edge.split(" ↔️ ")[0])
            nodes.add(edge.split(" ↔️ ")[1])
        elif "→" in edge:
            nodes.add(edge.split(" → ")[0])
            nodes.add(edge.split(" → ")[1])
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G, nodes, "labels")
    for edge in edges:
        if "↔️" in edge:
            G.add_edge(edge.split(" ↔️ ")[0], edge.split(" ↔️ ")[1])
            G.add_edge(edge.split(" ↔️ ")[1], edge.split(" ↔️ ")[0])
        elif "→" in edge:
            G.add_edge(edge.split(" → ")[0], edge.split(" → ")[1])
    st.session_state["graph"] = G        


if st.session_state["graph"]:
    G: nx.DiGraph = st.session_state["graph"]
    nodes = [n for n in G.nodes if n != "User_proxy"]
    st.session_state["codegpt_agents_list"] = [item for item in list(codegpt_system_messages.keys()) if item in nodes]
    st.session_state["other_agents_list"] = [item for item in list(st.session_state["other_agents"].keys()) if item in nodes]
    graph = graphviz.Digraph()
    graph.edges(G.edges)
    with col3:
        st.graphviz_chart(graph)
    st.write("-------")
    with st.sidebar.form("start"):
        user_task = st.text_area("User task")
        start = st.form_submit_button("Start")
        
    tab1, tab2 = st.tabs(["AutoGen", "Agent System Messages"])

    st.session_state["other_agents"]["Planner"] = (f"""You know nothing about the task, suggest a BRIEF plan to solve it. You cannot interact with User_proxy.
Explain the plan first. Be clear which step is performed by agents {', '.join(st.session_state["codegpt_agents_list"])}. These agents have the following system messages:
""" +  "\n".join([f"> {agent_name}: {codegpt_system_messages[agent_name]}" for agent_name in st.session_state["codegpt_agents_list"]])
)
    st.session_state["other_agents"]["PlanFollower"] = f"""Follow the plan step by step. Use \"NEXT: <Agent>, <Question>\" to assign the actual <Question> to <Agent>. If the plan has been completed, return the results along with \"TERMINATE\"."""
    #st.session_state["other_agents"]["Critic"] = f"""Your task is to criticize the plan to solve the user task. If you consider the plan not good enough, say that to the "Planner"."""
    with tab2:
        codegpt_df = pd.DataFrame.from_dict(codegpt_system_messages, orient="index", columns=["System Message"]).loc[[n for n in nodes if n in codegpt_system_messages]]
        st.write("### CodeGPT System Messages")
        st.dataframe(codegpt_df, use_container_width=True) 
        df = pd.DataFrame.from_dict(st.session_state["other_agents"], orient="index", columns=["System Message"]).loc[[n for n in nodes if n in st.session_state["other_agents"]]]
        st.write("### Other Agents (Editable)")
        edited_df = st.data_editor(df, use_container_width=True)
        #st.dataframe(df.loc[nodes], use_container_width=True)

    with tab1:
        def is_termination_msg(content) -> bool:
            have_content = content.get("content", None) is not None
            if have_content and "TERMINATE" in content["content"]:
                return True
            return False
        
        if start and user_task:
            if "User_proxy" not in st.session_state["graph"].nodes:
                st.error("Error: User_proxy not in nodes")
                st.stop()
            with st.spinner("Creating Agents..."):
                codegpt_assistant_agents =[]
                assistant_agents = []
                for agent_name, row in codegpt_df.iterrows():
                    codgpt_assistant = CodeGPTAssistantAgent(CODEGPT_API_KEY, agent_id=st.session_state["agent_list"][agent_name]["id"])
                    # proxy_agent = CodeGPTStandaloneAssistantAgent(codgpt_assistant, llm_config={"config_list": [{'model': 'gpt-4-1106-preview',
                    #                                                                                             'api_key': OPENAI_API_KEY},]})
                    codegpt_assistant_agents.append(codgpt_assistant)

                for agent_name, row in edited_df.iterrows():
                    _is_termination_msg = is_termination_msg if agent_name == "PlanFollower" else None
                    assistant = autogen.AssistantAgent(name=agent_name, system_message=row["System Message"],
                                                    is_termination_msg=_is_termination_msg,
                                                    llm_config={"config_list": [{'model': 'gpt-4-1106-preview',
                                                                                    'api_key': OPENAI_API_KEY},]})
                        
                    assistant_agents.append(assistant)
                

            user_proxy = autogen.UserProxyAgent(
                    name="User_proxy",
                    system_message="Terminator admin.",
                    code_execution_config=False,
                    is_termination_msg=is_termination_msg,
                    human_input_mode="NEVER")
            agents = codegpt_assistant_agents + assistant_agents + [user_proxy]

            group_chat = GraphGroupChat(
                agents=agents,  # Include all agents
                messages=[],
                max_round=10,
                graph=G
            )

            manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": [{'model': 'gpt-4-1106-preview',
                                                        'api_key': OPENAI_API_KEY}]})
            

            user_proxy.initiate_chat(manager, message=user_task)
            