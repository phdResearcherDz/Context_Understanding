from _Libreries.Ollama_API_Lib import *
from _Libreries._Trainer_Yes_No_Questions_Lib import load_dataset_yes_no

model_name = "llama3"
prompt_text = "Hi"
#
# prompt = '''
# Extract from the following text medical entities with there classes.
# the output need to be only entities with there medical classes in format of json as following:
# input = {INPUT}
# output = ?
# '''
prompt = '''
As part of my PhD research, I am working on improving context through data annotation. Specifically, I am interested in developing a prompt that can automatically identify and extract medical terms and abbreviations from a given text without altering the original content. The desired output should be in JSON format, structured as follows: { "Medical Terms":[], "Medical Abbreviations":[] }. This prompt aims to streamline the process of identifying and categorizing medical terms and abbreviations within text data for research purposes.
input = {INPUT}
output = ?
'''


elm_context = "CHecklist for critical Appraisal and data extraction for systematic Reviews of prediction Modelling Studies (CHARMS)., We will extract data based on the Checklist for Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modeling Studies (CHARMS), , Checklist for Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modeling Studies (CHARMS-PF)., Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modeling Studies (CHARMS), The CHARMS (critical appraisal and data extraction for systematic reviews of prediction modelling studies) checklist was created to provide methodological appraisals of predictive models, based on the best available scientific evidence and through systematic reviews., However, these models should be developed appropriately (CHecklist for critical Appraisal and data extraction for systematic Reviews of prediction Modeling Studies [CHARMS] and Prediction model Risk Of Bias ASsessment Tool [PROBAST] statements)., tudies with model updating. Data was extracted following the Checklist for critical Appraisal and data extraction for systematic Reviews of prediction Modelling Studies (CHARMS) checklist.PRIMARY AND SECONDARY OUTCOME MEA, Data collection was guided by the checklist for critical appraisal and data extraction for systematic reviews (CHARMS) and applicability and methodological quality assessment by the prediction model risk of bias assessment tool (PROBAST)., Studies were assessed using the checklist for critical appraisal and data extraction for systematic reviews of prediction modeling studies (CHARMS) checklist., methods in oncology. We used the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) statement, Prediction model Risk Of Bias ASsessment Tool (PROBAST) and CHecklist for critical Appraisal and data extraction for systematic Reviews of prediction Modelling Studies (CHARMS) to assess the methodological conduct of i, inclusion criteria). We followed the CHARMS recommendations (Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies), extracting the information from its 11 domains (Source of data, Critical appraisal and data extraction for systematic reviews of prediction modelling studies: the CHARMS checklist., The CHARMS (Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies) checklist was used for data extraction and quality assessment., ion making. Systematic reviews of prognostic model studies can help identify prognostic models that need to further be validated or are ready to be implemented in healthcare.OBJECTIVES: To provide a step-by-step guidance on how to conduct and read a systematic review of prognostic model studies and to provide an overview of methodology and guidance available for every step of the review progress.SOURCES: Published, peer-reviewed guidance articles.CONTENT: We describe the following steps for conducting a systematic review of prognosis studies: 1) Developing the review question using the Population, Index model, Comparator model, Outcome(s), Timing, Setting format, 2) Searching and selection of articles, 3) Data extraction using the Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies (CHARMS) checklist, 4) Quality and risk of bias assessment using the Prediction model Risk Of Bias ASsessment (PROBAST) tool, 5) Analysing data and undertaking quantitative meta-analysis, and 6) Presenting summary of findings, inte, te, or status of the publication. To carry out the systematic review, the CHARMS (Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies) guidel, The authors did not put any restrictions on the models included in their study regarding the model setting, prediction horizon or outcomes.Data extraction and synthesis Checklists of critical appraisal and data extraction for systematic reviews of prediction modelling studies (CHARMS) and prediction model risk of bias assessment tool (PROBAST) were used to guide developing of a standardised data extraction form., We critically appraised these models by means of criteria derived from the CHARMS (CHecklist for critical Appraisal and data extraction for systematic Reviews of prediction Modeling Studies) and PROBAST (Prediction model Risk Of Bias ASsessment Tool)., ist for critical Appraisal and data extraction for systematic Reviews of prediction Modeling Studies [CHARMS] and Prediction model Risk Of Bias ASsess, t for critical Appraisal and data extraction for systematic Reviews of prediction Modelling Studies (CHARMS) checklist.PRIMARY AND SECONDARY OUTCOME M, the CHARMS recommendations (Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies), extracting the information, The CHARMS (critical appraisal and data extraction for systematic reviews of prediction modelling studies) checklist was created to provide methodolog, ist for critical appraisal and data extraction for systematic reviews of prediction modeling studies (CHARMS) checklist. In total 89,959 citations wer, acted the data. We used the Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modeling Studies (CHARMS) checklist for the ri, ical appraisal and data extraction for systematic reviews of prediction modeling studies (CHARMS) and the prediction model risk of bias assessment too, nalysed (domains of CHARMS, Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies) the following: source of da, t for critical Appraisal and data extraction for systematic Reviews of prediction Modelling Studies (CHARMS) checklist.PRIMARY AND SECONDARY OUTCOME MEASURES: P, the CHARMS recommendations (Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies), extracting the information from its 11 domains (Source of data, Participants, etc). We determin, acted the data. We used the Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modeling Studies (CHARMS) checklist for the risk of , ical appraisal and data extraction for systematic reviews of prediction modeling studies (CHARMS) and the prediction model risk of bias assessment tool (PROBAST) were used for the data extraction process and critical appraisal.RESULTS: From the 29, nalysed (domains of CHARMS, Critical Appraisal and Data Extraction for Systematic Reviews of Prediction Modelling Studies) the following: source of data, participants, outcome to be predicted, candidate predictors, sample size, missing data, model development, model performance, model evaluation, results and interpretation and discussion.RESULTS: We found tw"

test_element_prompt = prompt.replace("{INPUT}", elm_context)
response = generate_chat_completion(model_name, test_element_prompt, stream=False)
print(response["message"]["content"])
#
# dataset = load_dataset_yes_no("Pre_Processed_Datasets/BioASQ/test.json")
# for question in dataset:
#
#     response = generate_chat_completion(model_name, chat_messages, stream=False)
#
