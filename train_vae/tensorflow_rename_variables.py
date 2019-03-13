# Adapted from https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
import getopt
import sys

import tensorflow as tf

usage_str = ('python tensorflow_rename_variables.py '
             '--checkpoint_dir=path/to/dir/ --replace_from=substr '
             '--replace_to=substr --add_prefix=abc --dry_run')
find_usage_str = ('python tensorflow_rename_variables.py '
                  '--checkpoint_dir=path/to/dir/ --find_str=[\'!\']substr')
comp_usage_str = ('python tensorflow_rename_variables.py '
                  '--checkpoint_dir=path/to/dir/ '
                  '--checkpoint_dir2=path/to/dir/')


def print_usage_str():
    print('Please specify a checkpoint_dir. Usage:')
    print('%s\nor\n%s\nor\n%s' % (usage_str, find_usage_str, comp_usage_str))
    print('Note: checkpoint_dir should be a *DIR*, not a file')


def compare(checkpoint_dir, checkpoint_dir2):
    import difflib
    with tf.Session():
        list1 = [el1 for (el1, el2) in
                 tf.contrib.framework.list_variables(checkpoint_dir)]
        list2 = [el1 for (el1, el2) in
                 tf.contrib.framework.list_variables(checkpoint_dir2)]
        for k1 in list1:
            if k1 in list2:
                continue
            else:
                print('{} close matches: {}'.format(
                    k1, difflib.get_close_matches(k1, list2)))


def find(checkpoint_dir, find_str):
    with tf.Session():
        negate = find_str.startswith('!')
        if negate:
            find_str = find_str[1:]
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            if negate and find_str not in var_name:
                print('%s missing from %s.' % (find_str, var_name))
            if not negate and find_str in var_name:
                print('Found %s in %s.' % (find_str, var_name))


def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            new_name = add_prefix + var_name

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))

            # Create the variable, potentially renaming it
            var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)


def main(argv):
    checkpoint_dir = None
    checkpoint_dir2 = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False
    find_str = None

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=',
                                               'replace_from=', 'replace_to=',
                                               'add_prefix=', 'dry_run',
                                               'find_str=',
                                               'checkpoint_dir2='])
    except getopt.GetoptError as e:
        print(e)
        print_usage_str()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--checkpoint_dir2':
            checkpoint_dir2 = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True
        elif opt == '--find_str':
            find_str = arg

    if not checkpoint_dir:
        print_usage_str()
        sys.exit(2)

    if checkpoint_dir2:
        compare(checkpoint_dir, checkpoint_dir2)
    elif find_str:
        find(checkpoint_dir, find_str)
    else:
        rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])