import numpy as np
from pathlib import Path
from sklearn.tree import _tree
import sys
sys.path.append('..')
from main import get_data
import os
from argparse import ArgumentParser
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

def tree_to_code(tree, feature_names, output_file=None):
    tree_ = tree.tree_
    feature_name = [feature_names[i] 
                    if i != _tree.TREE_UNDEFINED else "undefined!" 
                    for i in tree_.feature]
    if not output_file:
        print("def tree({}):".format(", ".join(feature_names)))
    else:
        open(output_file, 'a').write("def tree({}):\n".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if not output_file:
                print("{}if {} <= {}:".format(indent, name, threshold))
            else:
                open(output_file, 'a').write("{}if {} <= {}:\n".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            if not output_file:
                print("{}else:  # if {} > {}".format(indent, name, threshold))
            else:
                open(output_file, 'a').write("{}else:  # if {} > {}\n".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            if not output_file:
                print("{}return {}".format(indent, np.argmax(tree_.value[node])))
            else:
                open(output_file, 'a').write("{}return {}\n".format(indent, np.argmax(tree_.value[node])))

    recurse(0, 1)

def parse_config_path(path):
    config = Path(path).parts[-1]
    hypers = config.split('_')
    criterion = hypers[0]
    min_samples_split = int(hypers[2].split('=')[-1])
    imputer = hypers[4].split('=')[-1]
    sample = hypers[5].split('=')[-1]
    sample_rate = hypers[6].split('=')[-1]
    model_config = {
        'criterion': criterion,
        'min_samples_split': min_samples_split
    }
    data_config = {
        'imputer': imputer,
        'sample': sample,
        'sample_rate': sample_rate
    }
    return model_config, data_config

def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    model_config, data_config = parse_config_path(args.best_path)
    model = DecisionTreeClassifier(class_weight='balanced', **model_config)
    args.sample = data_config['sample']
    args.sample_rate = data_config['sample_rate']
    args.imputer = data_config['imputer']
    train_data, train_label, test_data, test_label = get_data(args)
    model.fit(train_data, train_label)
    
    feature_names = ["Attr%d" % i for i in range(train_data.shape[-1])]
    rule_text = export_text(model, feature_names=feature_names)
    rule_pic = export_graphviz(model, feature_names=feature_names, filled=True)
    with open(os.path.join(args.output_path, 'rule.txt'), 'w') as f:
        f.write(rule_text)
    with open(os.path.join(args.output_path, 'tree.dot'), 'w') as f:
        f.write(rule_pic)
    os.system('dot -Tpng %s -o %s' % (os.path.join(args.output_path, 'tree.dot'), os.path.join(args.output_path, 'tree.png')))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--best_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='./rule')
    args = parser.parse_args()

    main(args)