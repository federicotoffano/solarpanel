class LocalPaths(object):

    from uuid import getnode as get_mac
    mac = get_mac()

    if mac == 450157882930061:  # dave's Gpu
        main_path = "/home/dbrowne/Desktop/"
    if mac == 45015788293006:  # dave's laptop
        main_path = "C:/Users/dbrowne/Desktop/"
    if mac == 450157882930061:  # dave's home
        main_path = "C:/Users/dave/Desktop/"

    image_dir = main_path + 'Solar_imgs'
    scripts_dir = main_path + 'SolarPanel_code'
    image_dir_mono = image_dir + '/mono/'

    def __init__(self, loc = ''):
        None
