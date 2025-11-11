import numpy as np
import pandas as pd
import json
from IPython.display import HTML, display

def generate_power_law(inter_fn, cls_s, cls_num, img_max, img_min, many_threshold, verbose=False):

    
    if verbose: print(cls_s)

    cls_split_f = [f/100 for f in cls_s]
    cls_split = [int(cls_num * f) for f in cls_split_f]
    cls_split.extend([cls_num - sum(cls_split)])

    many_cls_num = inter_fn(np.linspace(0, 1, cls_split[0], endpoint=True), img_max, many_threshold + 1).astype(int).tolist()

    if verbose: print(f"num: {len(many_cls_num)}, size: {sum(many_cls_num)}, max: {max(many_cls_num)}, min: {min(many_cls_num)}")

    few_cls_num = inter_fn(np.linspace(0, 1, cls_split[1], endpoint=True), many_threshold, img_min).astype(int).tolist()

    if verbose: print(f"num: {len(few_cls_num)}, size: {sum(few_cls_num)}, max: {max(few_cls_num)}, min: {min(few_cls_num)}")

    img_num_per_cls = many_cls_num + few_cls_num

    if verbose: print(f"total size: {sum(img_num_per_cls)}")

    return img_num_per_cls

def side_by_side(*dfs):
    for i in range(0, len(dfs), 2):
        html = '<div style="display:flex">'
        for df, title in dfs[i:i+2]:
            html += '<div style="margin-right: 2em">'
            html += title
            html += df.to_html()
            html += '</div>'
        html += '</div>'
        display(HTML(html))
    
def read_json(path, verbose=False):
    with open(path, "r") as f:
        rt = json.load(f)
        if verbose:
            return rt
        else:
            return {k: v for k, v in rt.items() if k in ["accuracy", "many_acc", "med_acc", "few_acc"]}

def average_pd(pds):
    avg_pd = 0
    for pd in pds:
        avg_pd += pd
    avg_pd = avg_pd/len(pds)
    return round(avg_pd, 1)

def read_pd(results, keep=None):
    seed_results = {0: {}}

    for r in results:
        name = r.parent.name
        seed = name.split("_")[-1]

        if 'inat2018' in name or 'minicifar100':
            name_ix = 1
        else:
            name_ix = 2

        if seed.isdigit():
            seed = int(seed)
            name = "_".join(name.split("_")[name_ix:-1])
        else:
            seed = 0
            name = "_".join(name.split("_")[name_ix:])

        if keep is not None and not (keep in name):
            continue

        if seed > 0:
            continue

        js = read_json(r)
        seed_results[seed][name] = js
    
    pd_0 = pd.DataFrame(seed_results[0]).T.sort_index()
    return average_pd([pd_0])