import streamlit as st
import requests
import copy
import os
import autogen
from autogen import AssistantAgent


title = "AutoGen with CodeGPT Agents"
st.set_page_config(page_title=title, layout="wide")
st.title(title)
st.write("""**NOTE:**
- *You will require a simple **agent without** docs or prompt*
- *It assumes that the **name** of each CodeGPT agent describes what it does*
- *It assumes that the **prompt** of each CodeGPT agent describes the knowledge contrains*""")
st.sidebar.header(title)

# Margen a la izquierda
padding_left = 10
padding_right = 20
st.markdown(
        f"""<style>
        .appview-container .main .block-container{{
        padding-left: {padding_left}rem;
        padding-right: {padding_right}rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

CODEGPT_AUTOGEN_API_BASE = os.environ.get("CODEGPT_AUTOGEN_API_BASE", "http://127.0.0.1:8000/v1/autogen")
DEFAULT_SYSTEM_MESSAGE = "Your knowledge exclude any other information. If you are unsure about an answer, seek assistance from others."
SEED = 42

# HACK: use st.write instead of print
import builtins
def st_write_print(*args, **kwargs):
    args = [arg.replace("\033[33m", "**").replace("\033[0m", "**") for arg in args]
    st.write((' '.join(map(str, args))))
builtins.print = st_write_print

# Termination message detection
def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False

def get_default_config_dict(codegpt_api_key):
    config_dict = {
    "base_url": CODEGPT_AUTOGEN_API_BASE,
    "api_base": CODEGPT_AUTOGEN_API_BASE,
    "api_type": "open_ai",
    "api_key": codegpt_api_key
    }

    return config_dict

def get_default_llm_config_dict(config_list):
    llm_config = {
        "timeout": 60,
        "seed": SEED,
        "config_list": config_list,
        "temperature": 0
        }
    return llm_config

def create_autogen_agent_from_codegpt(agent, codegpt_api_key,
                                      system_message=DEFAULT_SYSTEM_MESSAGE):
    config_dict = get_default_config_dict(codegpt_api_key)
    config_list = [ {**config_dict, "model": agent["id"]}]
    llm_config = get_default_llm_config_dict(config_list)
    autogen_agent = AssistantAgent(
        name=agent["name"],
        llm_config=llm_config,
        system_message=system_message
    )
    return autogen_agent


def chat(autogen_agents, user_task, max_round=12, clear_cache=True):
    if clear_cache:
        autogen.Completion.clear_cache(seed=SEED)
    for agent in autogen_agents:
        agent.reset()

    global_message={"role": "system",
                    "content": "Everyone cooperate and help to solve the following task: " + user_task}
    groupchat = autogen.GroupChat(agents=autogen_agents, messages=[global_message], max_round=max_round)

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user.initiate_chat(manager, message=user_task)


CODEGPT_API_KEY = st.sidebar.text_input("CodeGPT API Key", type="password")
if not CODEGPT_API_KEY:
    st.sidebar.warning("Please enter your CodeGPT API key.", icon="⚠️")
    st.stop()

authorization_header = {"Authorization": f"Bearer {CODEGPT_API_KEY}"}
response = requests.get("https://api.codegpt.co/v1/agent", headers=authorization_header)
agent_list = {}

if response.status_code == 200:
    for agent in response.json():
        agent_list[agent["name"]] = agent
else:
    st.sidebar.error(f"Incorrect API Key")
    st.stop()

simple_agent_name = st.sidebar.selectbox("Simple Agent (no prompt or docs)", list(agent_list.keys()))
if agent_list[simple_agent_name]["prompt"]:
    st.sidebar.error("This agent has a prompt, select another one.")  
    st.stop()

simple_agent = agent_list[simple_agent_name]
with st.sidebar.form("my_form"):
    agents_no_simple = copy.deepcopy(agent_list)
    del agents_no_simple[simple_agent_name] # delete simple from the list
    agents_to_use = st.multiselect("Agents to use", list(agents_no_simple.keys()))

    task = st.text_area("Question or Task to solve", placeholder="Explain me the meaining of life")
    max_round = st.number_input("Max interaction rounds", value=8, min_value=3, max_value=15)
    submitted = st.form_submit_button("Start interaction")
    #use_planner = st.checkbox("Use a Planner agent", value=True)


if submitted:
    if not task:
        st.error("Please enter a task.")
        st.stop()

  

    config_dict = get_default_config_dict(CODEGPT_API_KEY)
    config_list = [ {**config_dict, "model": simple_agent["id"]}]
    llm_config = get_default_llm_config_dict(config_list)
    # Planner agent

    agents_prompts = ""
    for agent_name in agents_to_use:
        agents_prompts += "   - " + agent_name + ": " + agent_list[agent_name]["prompt"] + "\n"


    planner = AssistantAgent(
        name="Planner",
        llm_config=llm_config,
        system_message="You know nothing about the user question or task, suggest a BRIEF plan to solve it."
                f"\nExplain the plan first. Be clear which step is performed by {', '.join(agents_to_use)}. Then ask the Guider to continue the conversation."""
                )

    guider = AssistantAgent(
        name="Guider",
        llm_config=llm_config,
        system_message=f"Guide the conversation one step a time, telling to JUST ONE agent ({', '.join(agents_to_use)}) a question who has to ask."
        " These are the agent instructions:\n" + agents_prompts +
        f"\n Example: {agents_to_use[0]}, can you explain me something?"
                )
    
    # Checker agent
    checker = autogen.AssistantAgent(
    name="Checker",
    system_message="Verify if the task is finished. If it is, gather the relevant information from the conversation, then"
    " reply with that information followed by `TERMINATE`. If it is not, ask to the Guider to continue the conversation.",
    is_termination_msg=is_termination_msg,
    llm_config=llm_config
    )

    # User Proxy
    user = autogen.UserProxyAgent(
    name="Human",
    is_termination_msg=is_termination_msg,
    human_input_mode="NEVER", # if ALWAYS it will ask for a input
    system_message="A human who ask questions or give tasks. It interact with the Guider first.",
    code_execution_config=False,
    )

    list_of_autogen_agents = [user, planner, guider, checker]

    # Add CodeGPT agents
    for agent_name in agents_to_use:
        autogen_agent = create_autogen_agent_from_codegpt(agent_list[agent_name], CODEGPT_API_KEY)
        list_of_autogen_agents.append(autogen_agent)

    # Start the chat!
    with st.spinner("Generating conversation..."):
        stop_button = st.sidebar.button("STOP", type="primary")
        if stop_button:
            st.stop()
        with st.expander("Conversation", expanded=True):
            chat(list_of_autogen_agents, task, max_round=max_round)

    final_response = checker.last_message()["content"].replace("TERMINATE", "").strip()
    st.info(final_response, icon="ℹ️")

