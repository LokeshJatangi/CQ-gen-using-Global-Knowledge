
import os
import json
import gzip
import pickle
import pandas as pd
import itertools
import networkx as nx
import stanza 
import yake
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
final_df = pd.read_pickle("/home/aesicd_42/Desktop/tejas/final_df.pkl")

def build_schema(text, nlp_pipeline, kw_extractor):

    '''
    example usage:
    text = 'Is it possible to read using this product at night?'
    nlp_pipeline = stanfordnlp.Pipeline()
    kw_extractor = yake.KeywordExtractor(n=2)
    build_schema(text, nlp_pipeline, kw_extractor)
    '''

    # run a sent tokenizer

    # run the following for each sent

    # VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    VERB_TAGS = ['VB', 'VBG', 'VBZ']

    # run StanfordNLP pipeline
    doc = nlp_pipeline(text)
    sent  = doc.sentences[0]

    # all tokens + root
    tokens = ['root']
    tokens += [t.words[0].text.lower() for t in sent.tokens]

    # obtain all verbs and their indices
    verbs = []
    verb_indices = []
    for t in sent.tokens:
        if t.words[0].xpos in VERB_TAGS:
            verbs.append(t.words[0].text)
            verb_indices.append(int(t.words[0].id))

    # remove all verbs from keywords
    try:
        keywords = kw_extractor.extract_keywords(text)
    except Exception:
        keywords = []

    unigram_keywords_map = {}
    for kw in keywords:
        unigram_keywords_map[kw[0].lower()] = kw[0].lower().split()
    
    unigram_keywords = list(itertools.chain(*list(unigram_keywords_map.values())))

    # after this everything will be unigrams

    keywords_wo_verbs = [kw for kw in unigram_keywords if kw not in verbs]

    # obtain all keyword indices
    keyword_indices = [tokens.index(kw) for kw in keywords_wo_verbs if kw in tokens]

    # initialize dependency tree
    G = nx.DiGraph()
    for dep_edge in sent.dependencies:
        G.add_edge(int(dep_edge[0].id), int(dep_edge[2].id), relation=dep_edge[1])

    relation_edge_dict = nx.get_edge_attributes(G,'relation')

    schema = {}

    tuple_schema = []

    for v in verb_indices:
        # find a path from verb v to each keywords
        for k in keyword_indices:
            # we are finding shortest paths
            try:
                path = nx.shortest_path(G, source=v, target=k)
            except:
                # print('No path obtained')
                continue
                # check is path contains more than 2 nodes
            if len(path) > 2:
                # walk backward from the target to source
                for i, node in reversed(list(enumerate(path))):
                    if node in verb_indices:
                        # retrieve the first parent verb of the keyword
                        # obtain the relation with its child on the path
                        tuple_schema.append((tokens[v], tokens[k], relation_edge_dict[(node, path[i+1])], len(path)))
                        break
            else:
                # default case for an one-hop path between verb and keyword
                tuple_schema.append((tokens[v], tokens[k], relation_edge_dict[(v, k)], 2))

    # retain only the closest verb for each keyword
    for kw in keywords_wo_verbs:
        kw_tuples = [t for t in tuple_schema if t[1] == kw]
        if kw_tuples:
            final_tuple = min(kw_tuples, key = lambda t: t[1])  
            schema[kw] = final_tuple[:-1] # drop the path length
        else:
            schema[kw] = kw

    merged_schema = {}
    captured_uni_kw = []
    for kw, uni_kw in unigram_keywords_map.items():
        if uni_kw[0] in keywords_wo_verbs:
            if uni_kw[0] not in captured_uni_kw:
                # collect tuples based on the first unigram entry
                merged_schema[kw] = schema[uni_kw[0]]
                captured_uni_kw.extend(uni_kw)
    
    for kw, t in merged_schema.items():
        if isinstance(t, tuple):  
            merged_schema[kw] = (t[0], kw, t[-1])
        else:
            merged_schema[kw] = kw

    merged_schema = list(merged_schema.values())

    # default case, no schema, hence keep all keywords including verbs
    if len(tuple_schema) == 0:
        merged_schema = [kw[0].lower() for kw in keywords]

    merged_schema = list(set(merged_schema))

    return merged_schema

# build schema for product descriptions

def build_schema_from_desc(text, kw_extractor):
    # remove all verbs from keywords
    try:
        keywords = kw_extractor.extract_keywords(text)
    except Exception:
        keywords = []

    unigram_keywords_map = {}
    for kw in keywords:
        unigram_keywords_map[kw[0].lower()] = kw[0].lower().split()

    merged_schema = []

    captured_uni_kw = []
    for kw, uni_kw in unigram_keywords_map.items():
        if uni_kw[0] not in captured_uni_kw:
            # collect tuples based on the first unigram entry
            merged_schema.append(kw)
            captured_uni_kw.extend(uni_kw)
    
    return merged_schema

def build_schema_from_table(table):
    if table:
        return list(table.keys())
    else:
        return []

        
unique_list = final_df['assigned_cluster'].unique()
count_df =pd.DataFrame(final_df['assigned_cluster'].value_counts() )


stanza.download('en')
nlp = stanza.Pipeline('en')
kw_extractor = yake.KeywordExtractor(n=2)

save_to_file = True
cat_100 = []
for i in unique_list:
  if count_df['assigned_cluster'][i]<400:
    cat_100.append(i)

cat_100.reverse()
#cat_100=['Imported']
newlist=['eBook Readers &amp; Accessories',
 'Blank Video Media',
 'Cleaning & Repair',
 'Ethernet Cables',
 'Floppy & Tape Drives',
 'VCRs',
 'Office Electronics Accessories',
 'Cable Security Devices',
 'Switches',
 'External Zip Drives',
 'Cassette Players & Recorders',
 'Television &amp; Video',
 'Stereo System Components',
 'GPS, Finders &amp; Accessories',
 'Cord Management',
 'Remote Controls',
 'Modem Cables',
 'Cables & Interconnects',
 'Power Cables',
 'Network Cards',
 'Blank Media',
 'Portable CD Players',
 'Home Audio',
 'Center-Channel Speakers',
 'Bookshelf Speakers',
 'Repeaters',
 'Microcassette Recorders',
 'Telescope & Microscope Accessories',
 'Monitor Accessories',
 'SCSI Cables',
 'Serial Cables',
 'Media Storage & Organization',
 'Firewire Cables',
 'Satellite Speakers',
 'Outdoor Speakers',
 'Satellite TV Equipment',
 'Camcorder Accessories',
 'Backpacks']
cat_storage=[]
for cat in newlist : 
  prod_dict = {}
  cat_df = final_df[final_df['assigned_cluster'] == cat]
  grps = cat_df.groupby(cat_df['asin'])
  cat_storage.append(cat)
  print(cat)

  for name, g in grps:
      item_dict = {}
      qa = []
      item_id = g['asin'].iloc[0]
      for i, r in g.iterrows():
          item_dict['title'] = r['title']
          item_dict['category'] = r['assigned_cluster']
          # item_dict['description'] = r['description']
          item_dict['description_schema'] = build_schema_from_desc(' '.join(r['description']), kw_extractor)
          # item_dict['table1'] = r['tech1']
          # item_dict['table2'] = r['tech2']
          table_schema = []
          #table_schema.extend(build_schema_from_table(r['tech1']))
          #table_schema.extend(build_schema_from_table(r['tech2']))
          item_dict['table_schema'] = table_schema

          schema = build_schema(r['question'].lower(), nlp, kw_extractor)
          qa.append({'question': r['question'], 'schema': schema})
      item_dict['questions'] = qa
      
      prod_dict[item_id] = item_dict
      
  
  
  print('Category {} has {} items.'.format(cat, len(prod_dict)))
  if save_to_file:
          with open('{}_schema.json'.format(cat), 'w') as fp:
              json.dump(prod_dict, fp, indent=4)

