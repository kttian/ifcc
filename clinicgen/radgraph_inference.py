import os
import glob
import json 
import re
import argparse

uid = ""

def get_file_list(path):
    
    """Gets path to all the reports (.txt format files) in the specified folder, and
    saves it in a temporary json file
    
        Args:
            path: Path to the folder containing the reports
    """
    
    file_list = [item for item in glob.glob(f"{path}/**/*.txt", recursive=True)]
    
    # Number of files for inference at once depends on the memory available.
    ## Recemmended to use no more than batches of 25,000 files
    
    with open('./' + uid + '_temp_file_list.json', 'w') as f:
        json.dump(file_list, f)

def preprocess_reports():
    
    """ Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    
    file_list = json.load(open("./" + uid + "_temp_file_list.json"))
    final_list = []
    for idx, file in enumerate(file_list):

        temp_file = open(file).read()
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',temp_file).split()
        temp_dict = {}

        temp_dict["doc_key"] = file
        
        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

        if(idx % 1000 == 0):
            print(f"{idx+1} reports done")
    
    print(f"{idx+1} reports done")
    
    with open("./" + uid + "_temp_dygie_input.json",'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")
    return final_list

def run_inference(model_path, inputs):
    archive = load_archive(model_path)
    predictor = Predictor.from_archive(archive, 'dygie')
    ptm_indexer = PretrainedTransformerMismatchedIndexer(
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
        max_length=512)
    indexer = {"bert": ptm_indexer}
    reader = DyGIEReader(max_span_width = 5, token_indexers = indexer)
    results = []
    for doc_text in inputs:
        # Loop over the documents.
        instance = reader.text_to_instance(doc_text)
        result = predictor.predict_instance(instance)
        results.append(result)
    return results

def run_inference_old(model_path, cuda):
    
    """ Runs the inference on the processed input files. Saves the result in a
    temporary output file
    
    Args:
        model_path: Path to the model checkpoint
        cuda: GPU id
    
    
    """
    out_path =  "./" + uid + "_temp_dygie_output.json"
    data_path = "./" + uid + "_temp_dygie_input.json"
    
    print(f"allennlp predict {model_path} {data_path} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {cuda} \
            --silent")
    
    os.system(f"allennlp predict {model_path} {data_path} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {cuda} \
            --silent")

def postprocess_reports(results):
    final_dict = {}

    for result in results:
        postprocess_individual_report(result, final_dict)
    return final_dict

def postprocess_reports_old(file_name="./temp_dygie_output.json"):
    
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}

    # file_name = f"./" + uid + "_temp_dygie_output.json"
    data = []

    with open(file_name,'r') as f:
        for line in f:
            data.append(json.loads(line))

    for file in data:
        postprocess_individual_report(file, final_dict)
    
    return final_dict

def postprocess_individual_report(file, final_dict, data_source=None):
    
    """Postprocesses individual report
    
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    
    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n,r,s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = "inference"

        final_dict[file['doc_key']] = temp_dict
    
    except:
        print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")
        
def get_entity(n,r,s):
    
    """Gets the entities for individual reports
    
    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence
        
    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json 
    
    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx+1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i,val in enumerate(rel_list) if val== [start_idx, end_idx]]
        for i,val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab,str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx+1)] = temp_dict
    
    return dict_entity

def cleanup():
    """Removes all the temporary files created during the inference process
    
    """
    os.system("rm " + uid + "_temp_file_list.json")
    os.system("rm " + uid + "_temp_dygie_input.json")
    os.system("rm " + uid + "_temp_dygie_output.json")

def run(model_path, data_path, out_path, cuda):
    
    print("Getting paths to all the reports...", uid, flush=True)
    get_file_list(data_path)
    print(f"Got all the paths.", flush=True)
    
    print("Preprocessing all the reports...", flush=True)
    preprocess_reports()
    print("Done with preprocessing.", flush=True)
    
    print("Running the inference now... This can take a bit of time")
    run_inference(model_path, cuda)
    print("Inference completed.", flush=True)
    
    print("Postprocessing output file...", flush=True)
    final_dict = postprocess_reports()
    print("Done postprocessing.", flush=True)
    
    print("Saving results and performing final cleanup...", flush=True)
    cleanup()
    
    with open(out_path,'w') as outfile:
        json.dump(final_dict, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, nargs='?', required=True,
                        help='path to model checkpoint')
    
    parser.add_argument('--data_path', type=str, nargs='?', required=True,
                        help='path to folder containing reports')
    
    parser.add_argument('--out_path', type=str, nargs='?', required=True,
                        help='path to file to write results')
    
    parser.add_argument('--cuda_device', type=int, nargs='?', required=False,
                        default = -1, help='id of GPU, if to use')

    parser.add_argument('--uid', type=str, nargs='?', required=False,
                        default = -1, help='unique id of temp file')

    
    args = parser.parse_args()

    uid = args.uid
    
    run(args.model_path, args.data_path, args.out_path, args.cuda_device)
