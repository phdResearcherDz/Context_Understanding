from lxml import etree
import json

def process_xml_file(xml_file_path):
    # Parse the XML file
    tree = etree.parse(xml_file_path)
    root = tree.getroot()

    extracted_data = []
    for topic in root.xpath('.//topic[@t_name="Alzheimer"]'):
        for reading_test in topic.xpath('.//reading-test'):
            for doc in reading_test.xpath('.//doc'):
                context = doc.text.strip()
                for q in reading_test.xpath('.//q'):
                    question_text = q.find('.//q_str').text.strip()
                    answers = []
                    correct_answer = None
                    for answer in q.xpath('.//answer'):
                        answer_text = answer.text.strip()
                        answers.append(answer_text)
                        if answer.get('correct') == 'Yes':
                            correct_answer = answer_text

                    extracted_data.append({
                        "question": question_text,
                        "context": context,
                        "choices": answers,
                        "answer": correct_answer
                    })

    return extracted_data

# Specify the path to your XML file
xml_file_path = '../Datasets/QA4MRE-Alz/Data/QA4MRE_XML_Files-data/2013/2013_EN_GS.xml'

# Process the XML file and extract the data
extracted_data = process_xml_file(xml_file_path)

# Save the extracted data to a JSON file
with open("all_2013.json", "w") as outfile:
    json.dump(extracted_data, outfile, indent=4)

