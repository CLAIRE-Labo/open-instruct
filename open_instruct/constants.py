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
PHI2_CHAT_TEMPLATE = """{% if messages and messages[0]['role'] == 'system' %}
### System: {{ messages[0]['content'] }}

{% set messages = messages[1:] %}
{%- endif %}

{%- for message in messages %}
{%- if message['role'] == 'user' %}
### Human: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}
### Assistant: {{ message['content'] }}

{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}
### Assistant: {% endif %}"""