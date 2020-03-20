#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#session = vk.Session(access_token='4c7c952b4c7c952b4c7c952bf14c1a736544c7c4c7c952b179d212fd48229afe166e816')
#vk_api = vk.API(session)
# по умолчанию мой idшник вк(Денис)
def vkprint(*message_array, peer_id = '110014788', attachments = ''):
    """
    Vk_api bot, which works like print(...) but also writes message_array into peer_id dialogue.
    To give him access, write in https://vk.com/pythonreminder messages and change peer_id in properties.
    It returns the number of messages in dialogue
    TBD: export pictures in vk dialogue
    """
    #message = [i for i in message.replace(',', ' ').split()]
    import vk
    python_bot_session = vk.Session(access_token='a8dede3a2d1a42fa9bc495db9b437ca55671824c985d1faa77371943a6d36b5508021a74cadc36b645cae')
    python_bot = vk.API(python_bot_session)
    
    #peer_id = peer_id
    message_list =[i for i in message_array]
    message = ''
    for i in message_list:
        message+=' '+ str(i)

    #attachments = ''
    # TBD: разобраться с Attachments
    print(message)
    return python_bot.messages.send(v = '5.89',user_id = peer_id, message = message, attachment = ','.join(attachments))
    

