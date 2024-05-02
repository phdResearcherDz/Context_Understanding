import requests


def testOnHuggingFaceAPI():
    API_URL = "https://api-inference.huggingface.co/models/dmis-lab/biobert-large-cased-v1.1-squad"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": {
            "question": "Which splicing factors have been associated with alternative splicing in PLN R14del hearts?",
            "context": "Our work suggests that an intricate interplay of programs controlling gene expression levels and AS is fundamental to organ development, especially for the brain and heart., Bioinformatical analysis pointed to the tissue-specific splicing factors Srrm4 and Nova1 as likely upstream regulators of the observed splicing changes in the PLN-R14del cardiomyocytes. "
        },
    })

    print(output)

