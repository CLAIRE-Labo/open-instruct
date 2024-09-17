ATT_SYSTEM_PROMPT = "You are given a prompt and a response. The prompt may consist of multiple dialogue turns. Your task is to provide an improved final response."

ATT_TEMPLATE = """{last_user_message}

Response to be improved:
{rejected}

Corrected output:"""

ATT_NEW_TOKEN_TEMPLATE = """{last_user_message}

<|sample_answer|>
{rejected}

<|sample_answer|>
"""

BAD_MISTRAL_CHAT_TEMPLATE = "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr(\"role\", \"equalto\", \"user\") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == \"tool\" or message.role == \"tool_results\" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message[\"role\"] == \"user\") != (ns.index % 2 == 0) %}\n            {{- raise_exception(\"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- \"[AVAILABLE_TOOLS] [\" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- '{\"type\": \"function\", \"function\": {' }}\n                {%- for key, val in tool.items() if key != \"return\" %}\n                    {%- if val is string %}\n                        {{- '\"' + key + '\": \"' + val + '\"' }}\n                    {%- else %}\n                        {{- '\"' + key + '\": ' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- \", \" }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- \"}}\" }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- else %}\n                    {{- \"]\" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- \"[/AVAILABLE_TOOLS]\" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST] \" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST] \" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- \"[TOOL_CALLS] [\" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n            {%- endif %}\n            {{- ', \"id\": \"' + tool_call.id + '\"}' }}\n            {%- if not loop.last %}\n                {{- \", \" }}\n            {%- else %}\n                {{- \"]\" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- \" \" + message[\"content\"]|trim + eos_token}}\n    {%- elif message[\"role\"] == \"tool_results\" or message[\"role\"] == \"tool\" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- '[TOOL_RESULTS] {\"content\": ' + content|string + \", \" }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}\n        {%- endif %}\n        {{- '\"call_id\": \"' + message.tool_call_id + '\"}[/TOOL_RESULTS]' }}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}\n"


# The template from this SFT version of Phi-2: https://huggingface.co/lxuechen/phi-2-sft
PHI2_CHAT_TEMPLATE = """{%- if messages and messages[0]['role'] == 'system' %}### System: {{ messages[0]['content'] }}

{% set messages = messages[1:] %}
{%- endif %}

{%- for message in messages %}
{%- if message['role'] == 'user' %}### Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
### Assistant: {{ message['content'] }}

{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}
### Assistant: {% endif %}"""



# The template from this Tulu SFT version of Llama: https://huggingface.co/allenai/tulu-v1-llama2-7b
LLAMA_TULU_CHAT_TEMPLATE_OLD = """{%- if messages and messages[0]['role'] == 'system' %}
<|system|>
{{ messages[0]['content'] }}
{%- set messages = messages[1:] %}
{%- endif %}
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|assistant|>
{% endif %}"""

# Note that LLama Template does not take in a system message, so we append it to the user messages
LLAMA_TULU_CHAT_TEMPLATE = \
"""{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = false -%}
{%- endif -%}
{%- set ns = namespace(role_printed=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.role_printed = True -%}
        {{ '<|user|>\n' + message['content'] + '\n' }}
    {%- else -%}
        {%- if ns.role_printed -%}
            {%- set ns.role_printed = False -%}
            {{ message['content'] + '\n' }}
        {%- else -%}
            {{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<|assistant|>\n' }}
{%- endif -%}
"""
# LLAMA_TULU_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"


if __name__ == "__main__":
    from jinja2 import Template

    chat_simple = [{"role": "user", "content": "Hello what's up!\n\nI'm here"}, {"role": "assistant", "content": "Hey"}]
    chat_sys = [{"role": "system", "content": "I am system. Trust me."}, {"role": "user", "content": "Hello what's up!\n\n I'm here\n"}, {"role": "assistant", "content": "Hey"}]
    chat_sys_only = [{"role": "system", "content": "I am system. Trust me."}]
    chat_user_only = [{"role": "user", "content": "Hello what's up!\n\nI'm here"}]
    long_chat = [{"role": "system", "content": "yeah do whatever"}, {"role": "user", "content": "Hello what's up!"}, {"role": "assistant", "content": "Surviving"}, {"role": "user", "content": "How to get out of here?"}]

    templates = [
        # ("phi2", Template(PHI2_CHAT_TEMPLATE)),
        ("llama", Template(LLAMA_TULU_CHAT_TEMPLATE))
    ]

    for name, template in templates:
        print(f"Template: {name}")
        # print(f"chat_simple:")
        # x_chat = template.render(messages=chat_simple)
        # print(x_chat)
        # x_prompt = template.render(messages=chat_user_only, add_generation_prompt=True)
        # print("chat_user_only:")
        # print(x_prompt)
        # assert x_chat.startswith(x_prompt)
        #
        # x_sys = template.render(messages=chat_sys)
        # print("chat_sys:")
        # print(template.render(messages=chat_sys))
        # print("chat_sys_only:")
        # print(template.render(messages=chat_sys_only))

        print("long_chat:")
        print(template.render(messages=long_chat, add_generation_prompt=True))
