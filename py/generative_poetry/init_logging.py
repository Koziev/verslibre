import logging
import datetime
import coloredlogs
import absl.logging  # https://github.com/tensorflow/tensorflow/issues/26691


def init_logging(logfile_path, debugging=True):
    # настраиваем логирование в файл и эхо-печать в консоль

    # заменяем подстроку {DATETIME} в имени файла лога на текущее время и дату, чтобы логи от новых
    # запусков ботов не затирались
    if '{DATETIME}' in logfile_path:
        logfile_path = logfile_path.replace('{DATETIME}', datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

    # https://github.com/tensorflow/tensorflow/issues/26691
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    log_level = logging.DEBUG if debugging else logging.ERROR
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger('')
    logger.setLevel(log_level)

    file_fmt = '%(asctime)-15s %(levelname)-7s %(name)-25s %(message)s'

    if logfile_path:
        lf = logging.FileHandler(logfile_path, mode='w')
        lf.setLevel(logging.DEBUG)
        formatter = logging.Formatter(file_fmt)
        lf.setFormatter(formatter)
        logging.getLogger('').addHandler(lf)

    if True:
        field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
        field_styles["asctime"] = {}
        level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
        level_styles["debug"] = {}
        coloredlogs.install(
            level=log_level,
            use_chroot=False,
            fmt=file_fmt,  #"%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
            level_styles=level_styles,
            field_styles=field_styles,
        )
