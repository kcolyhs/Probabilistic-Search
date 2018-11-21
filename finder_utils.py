DEBUG_LEVEL = 0


def set_debug_level(level):
    global DEBUG_LEVEL
    DEBUG_LEVEL = level


def debug_print(str, level):
    if DEBUG_LEVEL >= level:
        print(str)
        return True
    return False


def error_print(str, level):
    return debug_print('[ERROR]' + str, level)


def cell_dist(cell1, cell2):
    return (cell1[0]+cell1[1]-cell2[0]-cell2[1])
