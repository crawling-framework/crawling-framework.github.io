import vk


def send_message(text: str, vk_id: str):
    token = "483dc0ad761894c153481dec6e2fe3f2af38634cab54767cc7dde8966c581151873ffbf2b5bb1cb70a775"
    session = vk.Session(access_token=token)
    python_bot = vk.API(session)
    return python_bot.messages.send(v='5.50', user_id=vk_id, message=text)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Send message to specified vk id')
    parser.add_argument('-m', required=True, help='text')
    parser.add_argument('--id', required=True, help='VK id')

    args = parser.parse_args()
    send_message(text=args.m, vk_id=args.id)
    # print('sent')
    return True


if __name__ == '__main__':
    send_message('test', vk_id='11014788')
    main()
