import random
from autogen import GroupChat

class GraphGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round=10, graph=None, verbose=False):
        super().__init__(agents, messages, max_round)
        self.previous_speaker = None  # Keep track of the previous speaker
        self.graph = graph  # The graph depicting who are the next speakers available
        self.verbose = verbose
        
    def select_speaker(self, last_speaker, selector):       
        self.previous_speaker = last_speaker

        # Check if last message suggests a next speaker or termination
        last_message = self.messages[-1] if self.messages else None
        suggested_next = None
        
        if last_message:
            if 'NEXT:' in last_message['content']:
                suggested_next = last_message['content'].split('NEXT: ')[-1].strip()
                # Strip full stop and comma
                suggested_next = suggested_next.replace('.', '').replace(',', '')
                print(f"Suggested next speaker from the last message: {suggested_next}")
                    
            elif 'TERMINATE' in last_message['content']:
                try:
                    return self.agent_by_name('User_proxy')
                except ValueError:
                    print(f"agent_by_name failed suggested_next: {suggested_next}")
                
        # Debugging print for the current previous speaker
        if self.previous_speaker is not None and self.verbose:
            print('Current previous speaker: ' + self.previous_speaker.name)

        # Selecting first round speaker
        if self.previous_speaker is None and self.graph is not None:
            eligible_speakers = [agent for agent in self.agents if self.graph.nodes[agent.name].get('first_round_speaker', False)]
            if self.verbose:
                print(f'First round eligible speakers: {[speaker.name for speaker in eligible_speakers]}')

        # Selecting successors of the previous speaker
        elif self.previous_speaker is not None and self.graph is not None:
            eligible_speaker_names = [target for target in self.graph.successors(self.previous_speaker.name)]
            eligible_speakers = [agent for agent in self.agents if agent.name in eligible_speaker_names]
            if self.verbose:
                print(f'Eligible speakers based on previous speaker:  {eligible_speaker_names}')

        else:
            eligible_speakers = self.agents

        # Debugging print for the next potential speakers
        if self.verbose:
            print(f"Eligible speakers based on graph and previous speaker {self.previous_speaker.name if self.previous_speaker else 'None'}: {[speaker.name for speaker in eligible_speakers]}")

        # Three attempts at getting the next_speaker
        # 1. Using suggested_next if suggested_next is in the eligible_speakers.name
        # 2. Using LLM to pick from eligible_speakers, given that there is some context in self.message
        # 3. Random (catch-all)
        next_speaker = None
        
        if eligible_speakers:
            if self.verbose:
                print(f"Selecting from eligible speakers: {[speaker.name for speaker in eligible_speakers]}")
            # 1. Using suggested_next if suggested_next is in the eligible_speakers.name
            if suggested_next in [speaker.name for speaker in eligible_speakers]:
                if self.verbose:
                    print("suggested_next is in eligible_speakers")
                next_speaker = self.agent_by_name(suggested_next)
                
            else:
                msgs_len = len(self.messages)
                if self.verbose:
                    print(f"msgs_len is now {msgs_len}")
                if len(self.messages) > 1:
                    # 2. Using LLM to pick from eligible_speakers, given that there is some context in self.message
                    if self.verbose:
                        print(f"Using LLM to pick from eligible_speakers: {[speaker.name for speaker in eligible_speakers]}")
                    selector.update_system_message(self.select_speaker_msg(eligible_speakers))
                    _, name = selector.generate_oai_reply(self.messages + [{
                        "role": "system",
                        "content": f"Read the above conversation. Then select the next role from {[agent.name for agent in eligible_speakers]} to play. Only return the role.",
                    }])

                    # If exactly one agent is mentioned, use it. Otherwise, leave the OAI response unmodified
                    mentions = self._mentioned_agents(name, eligible_speakers)
                    if len(mentions) == 1:
                        name = next(iter(mentions))
                        next_speaker = self.agent_by_name(name)

                if next_speaker is None:
                    # 3. Random (catch-all)
                    next_speaker = random.choice(eligible_speakers)
                    
            if self.verbose:    
                print(f"Selected next speaker: {next_speaker.name}")

            return next_speaker
        else:
            # Cannot return next_speaker with no eligible speakers
            raise ValueError("No eligible speakers found based on the graph constraints.")