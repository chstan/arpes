import datetime
import errno
import os.path

from arpes.config import CONFIG, FIGURE_PATH


def path_for_plot(desired_path):
    workspace = CONFIG['WORKSPACE']
    assert(workspace is not None)

    filename = os.path.join(FIGURE_PATH, workspace,
                            datetime.date.today().isoformat(), desired_path)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    return filename


def path_for_holoviews(desired_path):
    skip_paths = ['.svg', '.png', '.jpeg', '.jpg', '.gif']

    prefix, ext = os.path.splitext(desired_path)

    if ext in skip_paths:
        return prefix

    return prefix + ext
