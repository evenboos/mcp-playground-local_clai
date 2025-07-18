import gradio as gr
import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import modelscope_studio.components.pro as pro
import modelscope_studio.components.antdx as antdx
import json
import queue
import threading
from voice_processor import VoiceProcessor

voice_queue = queue.Queue()
voice_processor = None
import whisper
from langchain.chat_models import init_chat_model
from exceptiongroup import ExceptionGroup
from ui_components.config_form import ConfigForm
from ui_components.mcp_servers_button import McpServersButton
from mcp_client import generate_with_mcp, get_mcp_prompts
from config import bot_config, default_mcp_config, default_mcp_prompts, default_mcp_servers, user_config, welcome_config, default_theme, default_locale, bot_avatars, primary_color, mcp_prompt_model
from env import api_key, internal_mcp_config, llm_api_key, llm_base_url, llm_model_name

# whisper_model = whisper.load_model("medium", device="cuda", download_root="./whisper_models")

# def transcribe_audio(audio_filepath):
#     if audio_filepath is None:
#         return ""
#     # 指定语言为简体中文，提高转录准确性
#     result = whisper_model.transcribe(audio_filepath, language='zh')
#     return result["text"]

def merge_mcp_config(mcp_config1, mcp_config2):
    return {
        "mcpServers": {
            **mcp_config1.get("mcpServers", {}),
            **mcp_config2.get("mcpServers", {})
        }
    }


def safe_json_loads(json_str_or_dict):
    if isinstance(json_str_or_dict, dict):
        return json_str_or_dict
    try:
        return json.loads(json_str_or_dict)
    except (json.JSONDecodeError, TypeError):
        return {}


def format_messages(messages):
    formatted_messages = []
    for message in messages:
        if message["role"] == "user":
            formatted_messages.append({
                "role": "user",
                "content": message["content"]
            })
        elif message["role"] == "assistant":
            formatted_messages.append({
                "role":
                "assistant",
                "content":
                "\n".join([
                    content["content"] for content in message["content"]
                    if content["type"] == "text"
                ])
            })
    return formatted_messages

def handle_voice_input(text_input, status_tag):
    try:
        item = voice_queue.get_nowait()
        if item["type"] == "text":
            text_input = item["text"]
            return gr.update(value=text_input), gr.update(value="识别成功", color="green")
        elif item["type"] == "status":
            return gr.update(), gr.update(value=item["text"], color=item["color"])
        elif item["type"] == "submit":
            # Trigger the submit function
            return gr.update(value=item["text"], trigger=True), gr.update(value="识别成功", color="green")
    except queue.Empty:
        pass
    return gr.update(), gr.update()

def start_voice_processor():
    global voice_processor
    voice_processor = VoiceProcessor(voice_queue)
    # Use a daemon thread to ensure it exits when the main app exits
    threading.Thread(target=voice_processor.start, daemon=True).start()


async def submit(input_value, config_form_value, mcp_config_value,
                 mcp_servers_btn_value, chatbot_value):
    model = config_form_value.get("model", "")
    sys_prompt = config_form_value.get("sys_prompt", "")
    
    # 确定是否使用本地模型
    use_local_model = model.startswith("local/")

    enabled_mcp_servers = [
        item["name"] for item in mcp_servers_btn_value["data_source"]
        if item.get("enabled") and not item.get("disabled")
    ]
    if input_value:
        chatbot_value.append({
            "role":
            "user",
            "content":
            input_value,
            "class_names":
            dict(content="user-message-content")
        })

    chatbot_value.append({
        "role": "assistant",
        "loading": True,
        "content": [],
        "header": model.split("/")[1],
        "avatar": bot_avatars.get(model, None),
        "status": "pending"
    })
    yield gr.update(
        loading=True, value=None), gr.update(disabled=True), gr.update(
            value=chatbot_value,
            bot_config=bot_config(
                disabled_actions=['edit', 'retry', 'delete']),
            user_config=user_config(disabled_actions=['edit', 'delete']))
    try:
        prev_chunk_type = None
        tool_name = ""
        tool_args = ""
        tool_content = ""
        async for chunk in generate_with_mcp(
                format_messages(chatbot_value[:-1]),
                mcp_config=merge_mcp_config(safe_json_loads(mcp_config_value),
                                            internal_mcp_config),
                enabled_mcp_servers=enabled_mcp_servers,
                sys_prompt=sys_prompt,
                get_llm=lambda: init_chat_model(
                    model=llm_model_name if use_local_model else model,
                    model_provider="openai",
                    api_key=llm_api_key if use_local_model else api_key,
                    base_url=llm_base_url if use_local_model else "https://api-inference.modelscope.cn/v1/")):
            chatbot_value[-1]["loading"] = False
            current_content = chatbot_value[-1]["content"]

            if prev_chunk_type != chunk["type"] and not (
                    prev_chunk_type == "tool_call_chunks"
                    and chunk["type"] == "tool"):
                current_content.append({})
            prev_chunk_type = chunk["type"]
            if chunk["type"] == "content":
                current_content[-1]['type'] = "text"
                if not isinstance(current_content[-1].get("content"), str):
                    current_content[-1]['content'] = ''
                current_content[-1]['content'] += chunk['content']
            elif chunk["type"] == "tool":
                if not isinstance(current_content[-1].get("content"), str):
                    current_content[-1]['content'] = ''
                chunk_content = chunk["content"]
                current_content[-1]["content"] = current_content[-1][
                    "content"] + f'\n\n**🎯 结果**\n```\n{chunk_content}\n```'
                tool_name = ""
                tool_args = ""
                tool_content = ""
                current_content[-1]['options']["status"] = "done"
            elif chunk["type"] == "tool_call_chunks":
                current_content[-1]['type'] = "tool"
                current_content[-1]['editable'] = False
                current_content[-1]['copyable'] = False
                if not isinstance(current_content[-1].get("options"), dict):
                    current_content[-1]['options'] = {
                        "title": "",
                        "status": "pending"
                    }
                if chunk["next_tool"]:
                    tool_name += ' '
                    tool_content = tool_content + f"**📝 参数**\n```json\n{tool_args}\n```\n\n"
                    tool_args = ""
                if chunk["name"]:
                    tool_name += chunk["name"]
                    current_content[-1]['options'][
                        "title"] = f"**🔧 调用 MCP 工具** `{tool_name}`"
                if chunk["content"]:
                    tool_args += chunk["content"]
                    current_content[-1][
                        'content'] = tool_content + f"**📝 参数**\n```json\n{tool_args}\n```"

            yield gr.skip(), gr.skip(), gr.update(value=chatbot_value)
    except ExceptionGroup as eg:
        e = eg.exceptions[0]
        chatbot_value[-1]["loading"] = False
        chatbot_value[-1]["content"] += [{
            "type":
            "text",
            "content":
            f'<span style="color: var(--color-red-500)">{str(e)}</span>'
        }]
        print('Error: ', e)
        raise gr.Error(str(e))
    except Exception as e:
        chatbot_value[-1]["loading"] = False
        chatbot_value[-1]["content"] += [{
            "type":
            "text",
            "content":
            f'<span style="color: var(--color-red-500)">{str(e)}</span>'
        }]
        print('Error: ', e)
        raise gr.Error(str(e))
    finally:
        chatbot_value[-1]["status"] = "done"
        yield gr.update(loading=False), gr.update(disabled=False), gr.update(
            value=chatbot_value,
            bot_config=bot_config(),
            user_config=user_config())


def cancel(chatbot_value):
    chatbot_value[-1]["loading"] = False
    chatbot_value[-1]["status"] = "done"
    chatbot_value[-1]["footer"] = "对话已暂停"
    yield gr.update(loading=False), gr.update(disabled=False), gr.update(
        value=chatbot_value,
        bot_config=bot_config(),
        user_config=user_config())


async def retry(config_form_value, mcp_config_value, mcp_servers_btn_value,
                chatbot_value, e: gr.EventData):
    index = e._data["payload"][0]["index"]
    chatbot_value = chatbot_value[:index]

    async for chunk in submit(None, config_form_value, mcp_config_value,
                              mcp_servers_btn_value, chatbot_value):
        yield chunk


def clear():
    return gr.update(value=None)


def select_welcome_prompt(e: gr.EventData):
    return gr.update(value=e._data["payload"][0]["value"]["description"])


def select_model(e: gr.EventData):
    return gr.update(visible=e._data["payload"][1].get("thought", False))


async def reset_mcp_config(mcp_servers_btn_value):
    mcp_servers_btn_value["data_source"] = default_mcp_servers
    return gr.update(value=default_mcp_config), gr.update(
        value=mcp_servers_btn_value), gr.update(
            welcome_config=welcome_config(default_mcp_prompts)), gr.update(
                value={
                    "mcp_config": default_mcp_config,
                    "mcp_prompts": default_mcp_prompts,
                    "mcp_servers": default_mcp_servers
                })


def has_mcp_config_changed(old_config: dict, new_config: dict) -> bool:
    old_servers = old_config.get("mcpServers", {})
    new_servers = new_config.get("mcpServers", {})

    if set(old_servers.keys()) != set(new_servers.keys()):
        return True

    for server_name in old_servers:
        old_server = old_servers[server_name]
        new_server = new_servers.get(server_name)
        if new_server is None:
            return True

        if old_server.get("type") == "sse" and new_server.get("type") == "sse":
            if old_server.get("url") != new_server.get("url"):
                return True
        else:
            return True
    return False


def save_mcp_config_wrapper(initial: bool):

    async def save_mcp_config(mcp_config_value, mcp_servers_btn_value,
                              browser_state_value):
        mcp_config = json.loads(mcp_config_value)
        prev_mcp_config = json.loads(browser_state_value["mcp_config"])
        browser_state_value["mcp_config"] = mcp_config_value
        if has_mcp_config_changed(prev_mcp_config, mcp_config):
            mcp_servers_btn_value["data_source"] = [{
                "name": mcp_name,
                "enabled": True
            } for mcp_name in mcp_config.get("mcpServers", {}).keys()
                                                    ] + default_mcp_servers
            browser_state_value["mcp_servers"] = mcp_servers_btn_value[
                "data_source"]
            yield gr.update(
                welcome_config=welcome_config({}, loading=True)), gr.update(
                    value=mcp_servers_btn_value), gr.skip()
            if not initial:
                gr.Success("保存成功")
            prompts = await get_mcp_prompts(
                mcp_config=merge_mcp_config(mcp_config, internal_mcp_config),
                get_llm=lambda: init_chat_model(
                    model=llm_model_name,
                    model_provider="openai",
                    api_key=llm_api_key,
                    base_url=llm_base_url))

            browser_state_value["mcp_prompts"] = prompts
            yield gr.update(
                welcome_config=welcome_config(prompts)), gr.skip(), gr.update(
                    value=browser_state_value)
        else:
            yield gr.skip(), gr.skip(), gr.update(value=browser_state_value)
            if not initial:
                gr.Success("保存成功")

    return save_mcp_config


def save_mcp_servers(mcp_servers_btn_value, browser_state_value):
    browser_state_value["mcp_servers"] = mcp_servers_btn_value["data_source"]
    return gr.update(value=browser_state_value)


def load(mcp_servers_btn_value, browser_state_value, url_mcp_config_value):
    if browser_state_value:
        mcp_servers_btn_value["data_source"] = browser_state_value[
            "mcp_servers"]
        try:
            url_mcp_config = json.loads(url_mcp_config_value)
        except:
            url_mcp_config = {}
        return gr.update(value=json.dumps(
            merge_mcp_config(json.loads(browser_state_value["mcp_config"]),
                             url_mcp_config),
            indent=4)), gr.update(welcome_config=welcome_config(
                browser_state_value["mcp_prompts"])), gr.update(
                    value=mcp_servers_btn_value)
    elif url_mcp_config_value:
        try:
            url_mcp_config = json.loads(url_mcp_config_value)
        except:
            url_mcp_config = {}
        return gr.update(value=json.dumps(merge_mcp_config(url_mcp_config, {}),
                                          indent=4)), gr.skip(), gr.skip()
    return gr.skip()


def lighten_color(hex_color, factor=0.2):
    hex_color = hex_color.lstrip("#")

    # 解析RGB值
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # 向白色方向调整
    r = min(255, r + int((255 - r) * factor))
    g = min(255, g + int((255 - g) * factor))
    b = min(255, b + int((255 - b) * factor))

    # 转回十六进制
    return f"{r:02x}{g:02x}{b:02x}"


lighten_primary_color = lighten_color(primary_color, 0.4)

css = f"""
.ms-gr-auto-loading-default-antd {{
    z-index: 1001 !important;
}}

.user-message-content {{
    background-color: #{lighten_primary_color};
}}
"""

with gr.Blocks(css=css) as demo:
    browser_state = gr.BrowserState(
        {
            "mcp_config": default_mcp_config,
            "mcp_prompts": default_mcp_prompts,
            "mcp_servers": default_mcp_servers
        },
        storage_key="mcp_config")

    with ms.Application(), antdx.XProvider(
            locale=default_locale, theme=default_theme), ms.AutoLoading():

        with antd.Badge.Ribbon(placement="start"):
            with ms.Slot("text"):
                with antd.Typography.Link(elem_style=dict(color="#fff"),
                                          type="link",
                                          href="https://modelscope.cn/mcp",
                                          href_target="_blank"):
                    with antd.Flex(align="center",
                                   gap=2,
                                   elem_style=dict(padding=2)):
                        antd.Icon("ExportOutlined",
                                  elem_style=dict(marginRight=4))
                        ms.Text("前往")
                        antd.Image("./assets/modelscope-mcp.png",
                                   preview=False,
                                   width=20,
                                   height=20)
                        ms.Text("MCP 广场")
            with antd.Flex(justify="center", gap="small", align="center"):
                antd.Image("./assets/logo.png",
                           preview=False,
                           elem_style=dict(backgroundColor="#fff"),
                           width=50,
                           height=50)
                antd.Typography.Title("MCP Playground",
                                      level=1,
                                      elem_style=dict(fontSize=28, margin=0))

        with antd.Tabs():
            with antd.Tabs.Item(label="实验场"):
                with antd.Flex(vertical=True,
                               gap="middle",
                               elem_style=dict(height=1000, maxHeight="75vh")):
                    with antd.Card(
                            elem_style=dict(flex=1,
                                            height=0,
                                            display="flex",
                                            flexDirection="column"),
                            styles=dict(body=dict(flex=1,
                                                  height=0,
                                                  display='flex',
                                                  flexDirection='column'))):
                        with antd.Alert(
                            message="MCP工具使用指南",
                            description="点击下方输入框左侧的工具图标，选择需要使用的MCP工具。已配置：Memory、Doc-Tool和Filesystem工具。",
                            type="info",
                            closable=True,
                            elem_style=dict(marginBottom="10px")
                        ):
                            pass
                        chatbot = pro.Chatbot(
                            height=0,
                            bot_config=bot_config(),
                            user_config=user_config(),
                            welcome_config=welcome_config(default_mcp_prompts),
                            elem_style=dict(flex=1))
                    with antdx.Sender() as input:
                        with ms.Slot("prefix"):
                            with antd.Flex(gap="small", align="center"):
                                voice_switch = antd.Switch(value=False)
                                voice_status_tag = antd.Tag("休眠中", color="gray")
                            with antd.Button(value=None,
                                             variant="text",
                                             color="default") as clear_btn:
                                with ms.Slot("icon"):
                                    antd.Icon("ClearOutlined")
                            mcp_servers_btn = McpServersButton(
                                data_source=default_mcp_servers)

            with antd.Tabs.Item(label="配置"):
                with antd.Flex(vertical=True, gap="small"):
                    with antd.Card():
                        config_form, mcp_config_confirm_btn, reset_mcp_config_btn, mcp_config = ConfigForm(
                        )

    url_mcp_config = gr.Textbox(visible=False)
    load_event = demo.load(
        fn=load,
        js=
        "(mcp_servers_btn_value, browser_state_value) => [mcp_servers_btn_value, browser_state_value, decodeURIComponent(new URLSearchParams(window.location.search).get('studio_additional_params') || '') || null]",
        inputs=[mcp_servers_btn, browser_state, url_mcp_config],
        outputs=[mcp_config, chatbot, mcp_servers_btn]).then(
            fn=start_voice_processor
        )
    
    gr.Timer(0.1).tick(
        fn=handle_voice_input,
        inputs=[input],
        outputs=[input, voice_status_tag]
    )

    voice_switch.change(
        fn=lambda x: voice_processor.start() if x else voice_processor.stop(),
        inputs=[voice_switch]
    )

    chatbot.welcome_prompt_select(fn=select_welcome_prompt,
                                  outputs=[input],
                                  queue=False)
    retry_event = chatbot.retry(
        fn=retry,
        inputs=[config_form, mcp_config, mcp_servers_btn, chatbot],
        outputs=[input, clear_btn, chatbot])
    clear_btn.click(fn=clear, outputs=[chatbot], queue=False)
    mcp_servers_btn.change(fn=save_mcp_servers,
                           inputs=[mcp_servers_btn, browser_state],
                           outputs=[browser_state])

    load_success_save_mcp_config_event = load_event.success(
        fn=save_mcp_config_wrapper(initial=True),
        inputs=[mcp_config, mcp_servers_btn, browser_state],
        outputs=[chatbot, mcp_servers_btn, browser_state])
    save_mcp_config_event = mcp_config_confirm_btn.click(
        fn=save_mcp_config_wrapper(initial=False),
        inputs=[mcp_config, mcp_servers_btn, browser_state],
        cancels=[load_success_save_mcp_config_event],
        outputs=[chatbot, mcp_servers_btn, browser_state])
    reset_mcp_config_btn.click(
        fn=reset_mcp_config,
        inputs=[mcp_servers_btn],
        outputs=[mcp_config, mcp_servers_btn, chatbot, browser_state],
        cancels=[save_mcp_config_event, load_success_save_mcp_config_event])
    submit_event = input.submit(
        fn=submit,
        inputs=[input, config_form, mcp_config, mcp_servers_btn, chatbot],
        outputs=[input, clear_btn, chatbot])
    input.cancel(fn=cancel,
                 inputs=[chatbot],
                 outputs=[input, clear_btn, chatbot],
                 cancels=[submit_event, retry_event],
                 queue=False)

demo.queue(default_concurrency_limit=100, max_size=100).launch(ssr_mode=False,
                                                               max_threads=100)
