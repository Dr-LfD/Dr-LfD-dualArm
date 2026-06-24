### TO HANG: my DASHSCOPE_API_KEY='sk-75486571d38547f8b43a15b7acd4b409'
### read hdf5 file, and starting from demo0, get the skill info, and replay the agentview. 
import h5py
import networkx as nx
import cv2
import json
from itertools import product

def get_sg(hdf5_group, sg_name):
    sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
    if sg_json is None:
        return None
    sg_str = sg_json.decode('utf-8')
    sg = nx.node_link_graph(json.loads(sg_str))
    return sg

def cv2_show_img(img, skill_name, demo_id, draw_rec = False):

    img = cv2.resize(img, (128*5, 128*5), interpolation=cv2.INTER_NEAREST)

    ## print the skill name on img
    cv2.putText(img, skill_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    ## print the demo id at bottom
    cv2.putText(img, f'demo_{demo_id}', (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if draw_rec:
        ## red boundary
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 5)

    cv2.imshow('img', img)
    ## sleep for 0.05 seconds
    cv2.waitKey(50)


def get_all_changes(sg_info):
    def get_changed_edges(sg1, sg2):
        if sg1 is None or sg2 is None:
            return set()
        edges1 = set(map(frozenset, sg1.edges()))
        edges2 = set(map(frozenset, sg2.edges()))
        return edges1.symmetric_difference(edges2)

    all_edges = set()
    all_sgs = {}
    for skill_name, skill_info in sg_info.items():
        pre_sg = get_sg(skill_info, 'pre_sg')
        cur_sg = get_sg(skill_info, 'cur_sg')
        eff_sg = get_sg(skill_info, 'eff_sg')

        changed_edges = get_changed_edges(pre_sg, cur_sg)
        changed_edges = changed_edges.union(get_changed_edges(cur_sg, eff_sg))
        
        all_edges = all_edges.union(changed_edges)

        for sg in [pre_sg, cur_sg, eff_sg]:
            if sg is None:
                continue
            all_sgs[sg.name] = sg
    # all_edges = set(map(set, all_edges))
    all_edges = list(map(tuple, all_edges))
    return all_edges, all_sgs

def get_all_sgs(sg_info):
    all_sgs = {}
    # all_robots = set()
    # all_objs = set()
    for skill_name, skill_info in sg_info.items():
        pre_sg = get_sg(skill_info, 'pre_sg')
        cur_sg = get_sg(skill_info, 'cur_sg')
        eff_sg = get_sg(skill_info, 'eff_sg')
        

        # rbts_utf8 = [rbt.decode('utf-8') for rbt in skill_info['related_rbts']]
        # objs_utf8 = [obj.decode('utf-8') for obj in skill_info['related_objs']]
        # all_robots = all_robots.union(set(rbts_utf8))
        # all_objs = all_objs.union(set(objs_utf8))
            
        # all_edges = list(product(all_robots, all_objs))

        for sg in [pre_sg, cur_sg, eff_sg]:
            if sg is None:
                continue
            all_sgs[sg.name] = sg

    all_edges = set()
    for sg in all_sgs.values():
        for edge in sg.edges():
            if edge not in all_edges and (edge[1], edge[0]) not in all_edges:
                all_edges.add(edge)

    return all_sgs
    
    

def replay_hdf5_file(args):
    hdf5_file_path = args.hdf5_file_path
    start_demo_id = args.start_demo_id
    end_demo_id = args.end_demo_id

    f_in = h5py.File(hdf5_file_path, 'r')
    demos = list(f_in["data"].keys())
    inds = [int(demo.split('_')[-1]) for demo in demos]
    inds = sorted(inds)

    for demo_id in range(start_demo_id, end_demo_id):
        if demo_id not in inds:
            print(f'demo_{demo_id} not in the hdf5 file')
            continue

        sg_info = f_in[f'data/demo_{demo_id}/sg_info']

        demo_len = f_in[f'data/demo_{demo_id}/obs/agentview_image'].shape[0]
        # all_sgs, all_robots, all_objs = get_all_sgs(sg_info)
        all_edges, all_sgs = get_all_changes(sg_info)
        
        wrist_image = f_in[f'data/demo_{demo_id}/obs/robot0_eye_in_hand_image']

        ## if edge connected, then idx from this sg is positive. 
        ## NOTE: if there are 2 same objs, nn may be confused
        contact_positive_samples = {tuple(edge) :set() for edge in all_edges}
        for sg in all_sgs.values():
            for edge in all_edges:
                if sg.has_edge(edge[0], edge[1]):
                    contact_positive_samples[edge]= contact_positive_samples[edge].union(set(sg.graph['idx_list']))

        contact_negative_samples = {edge :set() for edge in all_edges}
        for edge in all_edges:
            contact_negative_samples[edge] = set(range(demo_len)) - contact_positive_samples[edge]

        # for edge, idx_set in contact_positive_samples.items():

        #     for i in idx_set:
        #         img = wrist_image[i]
        #         cv2_show_img(img, f'{edge[0]} {edge[1]}', demo_id, draw_rec=False)

        for skill_name in sg_info.keys():
            
            skill_info = sg_info[skill_name]

            ## get the agentview_image
            agentview_image = f_in[f'data/demo_{demo_id}/obs/agentview_image']

            if 'bimanual' in skill_name:
                pre_sg = get_sg(skill_info, 'pre_sg')
                extended_ids = pre_sg.graph['idx_list']
                essential_ids = []
            else:
                # pre_sg = get_sg(skill_info, 'pre_sg')
                # extended_ids = pre_sg.graph['idx_list']
                extended_ids = skill_info['extended_ids'][()]
                essential_ids = skill_info['essential_ids'][()]

            for i in extended_ids:
                img = agentview_image[i]
                cv2_show_img(img, skill_name, demo_id, draw_rec=i in essential_ids)
                

    f_in.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file_path', type=str,
        # default='/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/two_arm_threading_pc_instance_sg_100.hdf5'
        default='/home/user/yzchen_ws/imitation_learning/dexmimicgen/datasets/generated/two_arm_three_piece_assembly_pc_instance_sg_50.hdf5'

                        )
    parser.add_argument('--start_demo_id', type=int, default=0)
    parser.add_argument('--end_demo_id', type=int, default=50)
    args = parser.parse_args()
    replay_hdf5_file(args)