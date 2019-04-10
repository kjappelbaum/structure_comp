from structure_comp.remove_duplicates import RemoveDuplicates

rd_rmsd_graph = RemoveDuplicates.from_folder(
    'csd_mofs', method='rmsd_graph')

rd_rmsd_graph.run_filtering()

print('found {} duplicats'.format(rd_rmsd_graph.duplicates)) 

