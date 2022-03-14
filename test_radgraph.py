import unittest
import json

from clinicgen.eval import EntityMatcher

import time 

def load_entities2(cls, path):
    start_time = time.perf_counter()
    sentences, entities = {}, {}
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            did, sid = entry['id'].split(cls.ID_SEPARATOR)
            sid = int(sid)
            if did not in sentences:
                sentences[did] = {}
            sentences[did][sid] = entry['text'].lower()
            if did not in entities:
                entities[did] = {}
            for entity in entry['nes']:
                # if entity['type'] in target_types:
                s = entity['text'].lower()
                if s not in entities[did]:
                    entities[did][s] = [entity['type']]
                else:
                    entities[did][s].append(entity['type'])
    end_time = time.perf_counter()
    print("FINISHED LOADING STANZA FILE", end_time-start_time)
    return entities

def compare_radgraph_stanza():
    start_time = time.perf_counter()
    radgraph_path = 'radgraph_inference_out_report_key.json'
    stanza_path = 'mimic-cxr_ner.txt'
    with open(radgraph_path, "r") as f:
        radgraph = json.load(f)
    end_time = time.perf_counter()
    print("FINISHED LOADING RADGRAPH FILE", end_time - start_time)
    stanza = load_entities2(EntityMatcher, stanza_path)

    rid_intersect = radgraph.keys() & stanza.keys()
    print("radgraph reports", len(radgraph))
    print("stanza reports", len(stanza))
    print("intersection", len(rid_intersect))
    start_time = time.perf_counter()
    counts = {}

    tot_match = 0
    tot_r = 0
    tot_s = 0
    for rid in rid_intersect:
        temp = {}
        match = 0
        radgraph_ent = radgraph[rid]['entities']
        stanza_ent = stanza[rid]
        r_total = len(radgraph_ent)
        s_total = len(stanza_ent)
        for key in radgraph_ent:
            token = radgraph_ent[key]['tokens']
            if token in stanza_ent:
                match += 1
        
        temp['match'] = match 
        temp['radgraph'] = r_total
        temp['stanza'] = s_total 
        counts[rid] = temp 

        tot_match += match 
        tot_r += r_total 
        tot_s += s_total
    end_time = time.perf_counter()
    avg_match = float(tot_match)/len(rid_intersect)
    avg_r = float(tot_r)/len(rid_intersect)
    avg_s = float(tot_s)/len(rid_intersect)
    print("avg_match", avg_match, "avg_r", avg_r, "avg_s", avg_s)
    print("FINISHED COUNTING ENTITIES", end_time - start_time)
    return counts 

def score_comparison():
    counts = compare_radgraph_stanza()
    total = len(counts)
    avg_r_denom = 0
    avg_s_denom = 0
    for rid in counts:
        match = counts[rid]['match']
        r_total = counts[rid]['radgraph']
        s_total = counts[rid]['stanza']
        if r_total > 0:
            avg_r_denom += float(match)/r_total
        if s_total > 0:
            avg_s_denom += float(match)/s_total 
    avg_r_denom /= total 
    avg_s_denom /= total 
    return avg_r_denom, avg_s_denom

    
class TestEntityMatcher(unittest.TestCase):
    def setUp(self):
        print("set up start")
        start_time = time.perf_counter()
        sentences = {'1': {0: 'No pleural effusions.'}, '2': {0: 'Enlarged heart.'}}
        entities = {'1': {'pleural': [0], 'effusions': [0]}, '2': {'heart': [0]}}
        target_types = {'ANATOMY': True, 'OBSERVATION': True}
        self.matcher = EntityMatcher(sentences, entities, target_types)
        print("set up end", time.perf_counter() - start_time)

    def test_score(self):
        print("score start")
        start_time = time.perf_counter()
        rs = self.matcher.score(['1', '2'], ['No pleural effusion.', 'Normal heart size.'])
        self.assertEqual(rs[1][0], 0.5)
        self.assertEqual(rs[1][1], 1.0)
        print("score end", time.perf_counter() - start_time)
    
    def test_radgraph_score(self):
        print("radgraph score start")
        start_time = time.perf_counter()
        rs = self.matcher.score_radgraph(['1', '2'], ['No pleural effusion.', 'Normal heart size.'])
        print("radgraph score end", time.perf_counter() - start_time)

if __name__ == '__main__':
    unittest.main()
    # print("GETTING STARTED")
    # r_score, s_score = score_comparison()
    # print("r_score", r_score)
    # print("s_score", s_score)

# class TestRadGraph(unittest.TestCase):
#     # same as TestEntityMatcher in test_eval...
#     def setUp(self):
#         sentences = {'1': {0: 'No pleural effusions.'}, '2': {0: 'Enlarged heart.'}}
#         entities = {'1': {'pleural': [0], 'effusions': [0]}, '2': {'heart': [0]}}
#         target_types = {'ANATOMY': True, 'OBSERVATION': True}
#         self.matcher = EntityMatcher(sentences, entities, target_types)
    
#     def test_score(self):
#         rids = [56543992, 52026760]
#         hypos = ['there is little change and no evidence of acute cardiopulmonary disease. no pneumonia or vascular congestion',
#                 'Heart size is normal. Mediastinum is normal. The lungs are clear. There is no pleural effusion or pneumothorax Right paracardiac opacity most likely represent fat pad by giving the abnormality in status persistent cough correlation with chest CT might be considered .']
#         path_mini = '/n/data1/hms/dbmi/rajpurkar/lab/home/kt220/dygiepp/outputs/radgraph_inference_out_report_key_mini.json'
#         self.matcher.load_radgraph(path_mini)
