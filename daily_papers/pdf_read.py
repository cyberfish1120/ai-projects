"""
This example describes how to use the workflow interface to chat.
"""

import os
# Our official coze sdk for Python [cozepy](https://github.com/coze-dev/coze-py)
from cozepy import COZE_CN_BASE_URL

# Get an access_token through personal access token or oauth.
coze_api_token = 'ppat_wTFwEkqJcVslGw7cKimLfgwFmibm8Aagt0iLTpkz0DekVe888Gri96eoEDV8Qh8L'
# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
coze_api_base = COZE_CN_BASE_URL

from cozepy import Coze, TokenAuth, Message, ChatStatus, MessageContentType  # noqa

# Init the Coze client through the access_token.
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)

# Create a workflow instance in Coze, copy the last number from the web link as the workflow's ID.
# workflow_id = '7513571882814750747'
workflow_id = '7517842621692968969'

# Call the coze.workflows.runs.create method to create a workflow run. The create method
# is a non-streaming chat and will return a WorkflowRunResult class.
def paper_read(pdf_url):
    try:
        workflow = coze.workflows.runs.create(
            workflow_id=workflow_id,
            parameters={
                'input': pdf_url
            }
            # is_async=True
        )
        return workflow.data
    except:
        return None

if __name__ == '__main__':
    paper_read('http://arxiv.org/pdf/2506.15674v1')