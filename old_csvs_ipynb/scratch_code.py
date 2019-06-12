# def resample_to_match_terms(df_full, terms_to_match, check_col='content_min_clean', groupby_col='troll_tweet', group_dict={0:'Control',1:'Troll'}):
    
#     group_sizes = [['GroupCode','Group Name','Count']]    
#     # Get the size of each group
#     for k,v in group_dict.items():
#         group_sizes.append([k,v,sum(df_full[groupby_col]==k)])
        
#     df_groups = bs.list2df(group_sizes)#, index_col='GroupCode')
#     n_samples = min(df_groups['Count'])
#     groupcode_to_match = df_groups['GroupCode'].loc[df_groups['Count'].idxmin(axis=0)]
    
    
#     df_build_sample = pd.DataFarme()
#     df_sample_index = []
    
#     # Separate out group to be matched to count term frquencies 
#     group_to_match = df_full.groupby(groupby_col).get_group(groupcode_to_match)
#     list_match_freqs = [['Term','#','']] # get a list of how many of the term is present in the group-to-match
    
#     for term in terms_to_match:
#         # Check how many of the term is present in the smaller group to match
#         to_match_count = len(group_to_match[check_col].str.contains(exp))
#         list_match_freqs.append([term,to_match_count])
    
# #     return n_samples

# resample_to_match_terms(df_full, ['bird'])