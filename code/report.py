"""
Title: report.py
Description: Generate the report for evaluations.
Author: Yeol Ye, University of Chicago
"""


# *******************************************************************
# Generate Reports
# *******************************************************************
def generate_each_report(table, metrics):
    print('------------------------------------------------------')
    for x in metrics:
        max_ = table[x].max()
        print('{}: max = {:.2f}, models = {}'
              .format(x, max_,
                      table.loc[table[x] == max_]['model_type'].unique()))


def generate_whole_report(tables, metrics):
    print('The best classifiers under different metrics: \n')
    for i in range(len(tables)):
        print('The {}th 6 months:'.format(i))
        generate_each_report(tables[i], metrics)
        print()
