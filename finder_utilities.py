DEBUG_LEVEL = 0


def debug_print(str, level):
    if DEBUG_LEVEL >= level:
        print(str)
        return True
    return False


def error_print(str, level):
    return debug_print('[ERROR]' + str, level)
