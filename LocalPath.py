class LocalPaths(object):

    from uuid import getnode as get_mac
    mac = get_mac()
    print(mac)

    if mac == 450157882930061:  # dave's Gpu
        main_path = "/home/dbrowne/Desktop/"
    if mac == 45015788293006:  # dave's laptop
        main_path = "C:/Users/dbrowne/Desktop/"
    if mac == 450157882930061:  # dave's home
        main_path = "C:/Users/dave/Desktop/"
    if mac == 167132875593445:  # fede's mac
        main_path = "/Users/ftoffano/AppData/SolarPanel/"
    if mac == 18431769921280:  # fede's asus
        main_path = "E:/AppData/SolarPanel/"

    image_dir = main_path + 'Solar_imgs'
    scripts_dir = main_path + 'SolarPanel_code'
    image_dir_mono = image_dir + '/mono/'


