import numpy as np
np.random.seed(5)
import argparse

print('parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--num_items', type=int, default=1000, help='number of items')
parser.add_argument('--num_users', type=int, default=200, help='number of users')
parser.add_argument('--num_user_attr', type=int, default=30, help='number of user attributes')
parser.add_argument('--num_item_attr', type=int, default=100, help='number of item attributes')
parser.add_argument('--num_rules', type=int, default=40, help='number of rules')
parser.add_argument('--per_rule_sparsity', type=float, default=0.30, help='per rule sparsity')
parser.add_argument('--attribute_sparsity', type=float, default=0.20, help='attribute sparsity')
parser.add_argument('--data_dir', type=str, default='data/intersection_only/', help='data directory')
args = parser.parse_args()

num_items = args.num_items
num_users = args.num_users
num_user_attr = args.num_user_attr
num_item_attr = args.num_item_attr
num_rules = args.num_rules
per_rule_sparsity = args.per_rule_sparsity
attribute_sparsity = args.attribute_sparsity
data_dir = args.data_dir
##### Generate Attributes
# generate 50 users with 10 attributes
user_attr = np.random.rand(num_users, num_user_attr)
user_attr = (user_attr < attribute_sparsity).astype(int) # sparsify the user attributes
# generate 50 items with 10 attributes
item_attr = np.random.rand(num_items, num_item_attr)
item_attr = (item_attr < attribute_sparsity).astype(int) # sparsify the item attributes

##### Generate Rules
rules = []
for i in range(num_rules):
    # sample variable number of attributes from user atrributes
    num_sample = np.random.randint(1, 5) ### 4 is a design choice, need to thing more on this
    user_attr_idx = np.random.choice(range(num_user_attr), num_sample, replace=False).tolist()
    
    # sample variable number of attributes from item atrributes
    num_sample = np.random.randint(1, 5) ### 4 is a design choice, need to thing more on this
    item_attr_idx = np.random.choice(range(num_item_attr), num_sample, replace=False).tolist()

    attr_idx = (user_attr_idx, item_attr_idx)
    rules.append(attr_idx)


## Look up function for user & item pairs that satisfy the rules
    
### Only intersection and positive of the RV is considered
def rule_look_up(rule):
    user_rule = rule[0]
    item_rule = rule[1]

    # Find all users that have the attributes 1 in the idx that is specified in the rule
    user_list = []
    for i in range(num_users):
        user_i = np.array(user_attr[i])
        if (user_i[user_rule] == 1).all():
            user_list.append(i)
    # Find all items that have the attributes 1 in the idx that is specified in the rule
    item_list = []
    for i in range(num_items):
        item_i = np.array(item_attr[i])
        if (item_i[item_rule] == 1).all():
            item_list.append(i)
    return user_list, item_list

## Generate the user-vs-item co-watch matrix

user_item = np.zeros((num_users, num_items))
user_item_tuple = []
for rule in rules:
    user_list, item_list = rule_look_up(rule)
    for user in user_list:
        # sample 30 % of the items
        item_sampled_list = np.random.choice(item_list, int(len(item_list) * per_rule_sparsity), replace=False)
        for item in item_sampled_list:
            user_item[user][item] = 1.0


final_vocab_user = set(np.where(np.array(user_item) == 1.0)[0])
final_vocab_item = set(np.where(np.array(user_item) == 1.0)[1])
print(len(final_vocab_user), len(final_vocab_item))


### Save the data

def np2csv(np_array, filename, headers):
    positive_list = list(zip(*np.where(np.array(np_array) == 1.0)))
    # save the positive list as csv
    with open(filename, 'w') as f:
        f.write(headers[0]+ ',' + headers[1] + '\n')
        for ele in positive_list:
            if filename == data_dir + 'user_item.csv' or filename == data_dir + 'train.csv':
                f.write(str(ele[0]) + ',' + str(ele[1]) + '\n')
            elif filename == data_dir + 'user_attr.csv':
                if ele[0] in final_vocab_user:
                    f.write(str(ele[0]) + ',' + str(ele[1]) + '\n')
            elif filename == data_dir + 'item_attr.csv':
                if ele[0] in final_vocab_item:
                    f.write(str(ele[0]) + ',' + str(ele[1]) + '\n')

np2csv(item_attr, data_dir + 'item_attr.csv', headers=['item_id', 'attr_id'])
np2csv(user_attr, data_dir + 'user_attr.csv', headers=['user_id', 'attr_id'])
np2csv(user_item, data_dir + 'user_item.csv', headers=['user_id', 'item_id'])


## Test/ Train splits.


test_dict = {}
count = 0
for i, user_profile in enumerate(user_item):
    viewd_items = np.where(user_profile == 1.0)[0]
    if len(viewd_items) <= 10:
        continue
    # sample 20% of the items from that list
    test_items = np.random.choice(viewd_items, int(len(viewd_items) * 0.2), replace=False)
    # set the test items to 0
    user_profile[test_items] = 0.0
    test_dict[i] = test_items
    count += len(test_items)

np2csv(user_item, data_dir + 'train.csv', headers=['user_id', 'item_id'])

def save_test_file(test_dict):
    with open(data_dir + 'test.csv', 'w') as f:
        f.write('user_id,item_id\n')
        for user, items in test_dict.items():
            for item in items:
                f.write(str(user) + ',' + str(item) + '\n')
save_test_file(test_dict)

## Save rules for eval
def save_rules(rules):
    with open(data_dir + 'rules.csv', 'w') as f:
        f.write('user_attr_ids' + ',' + 'item_attr_ids' + '\n')
        for rule in rules:
            f.write(str(rule[0]) + ',' + str(rule[1]) + '\n')
save_rules(rules)
print('Done')

