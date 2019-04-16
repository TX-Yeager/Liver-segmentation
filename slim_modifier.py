def print_function(f):
    print "before ..."
    f()
    print "after ..."


@print_function
def func():
    print "func was called"