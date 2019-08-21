import datetime
import json
import os
import urllib.request

from pathlib import Path
from tqdm import tqdm_notebook
from typing import List, Optional


__all__ = ('get_full_notebook_information', 'get_notebook_name', 'generate_logfile_path',
           'get_recent_logs', 'get_recent_history', 'wrap_tqdm')


def wrap_tqdm(x, interactive=True, *args, **kwargs):
    if not interactive:
        return x

    return tqdm_notebook(x, *args, **kwargs)


def get_full_notebook_information() -> Optional[dict]:
    """
    Javascriptless method to get information about the current Jupyter sessions and the one matching
    this kernel.
    :return:
    """

    try:  # Respect those that opt not to use IPython
        from notebook import notebookapp
        import ipykernel
    except ImportError:
        return None

    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    servers = notebookapp.list_running_servers()
    for server in servers:
        try:
            passwordless = not server['token'] and not server['password']
            url = server['url']+'api/sessions' + ('' if passwordless else '?token={}'.format(server['token']))
            sessions = json.load(urllib.request.urlopen(url))
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return {'server': server, 'session': sess,}
        except:
            pass
    return None


def get_notebook_name() -> Optional[str]:
    """
    Gets the unqualified name of the running Jupyter notebook, if there is a Jupyter session
    not protected by password.

    As an example, if you were running a notebook called "Doping-Analysis.ipynb"
    this would return "Doping-Analysis".

    If no notebook is running for this kernel or the Jupyter session is password protected, we
    can only return None.
    :return:
    """

    jupyter_info = get_full_notebook_information()

    try:
        return jupyter_info['session']['notebook']['name'].split('.')[0]
    except (KeyError, TypeError):
        return None


def generate_logfile_path() -> Path:
    """
    Generates a time and date qualified path for the notebook log file.
    :return:
    """

    base_name = get_notebook_name() or 'unnamed'
    full_name = '{}_{}_{}.log'.format(
        base_name,
        datetime.date.today().isoformat(),
        datetime.datetime.now().time().isoformat().split('.')[0].replace(':', '-')
    )
    return Path('logs') / full_name


def get_recent_history(n_items=10) -> List[str]:
    try:
        import IPython
        ipython = IPython.get_ipython()

        return [l[-1] for l in list(ipython.history_manager.get_tail(n=n_items, include_latest=True))]
    except (ImportError, AttributeError):
        return ['No accessible history.']


def get_recent_logs(n_bytes=1000) -> List[str]:
    import arpes.config
    try:
        import IPython
        ipython = IPython.get_ipython()
        if arpes.config.CONFIG['LOGGING_STARTED']:
            logging_file = arpes.config.CONFIG['LOGGING_FILE']

            print(logging_file)
            with open(logging_file, 'rb') as file:
                try:
                    file.seek(-n_bytes, os.SEEK_END)
                except OSError:
                    file.seek(0)

                lines = file.readlines()

            # ensure we get the most recent information
            final_cell = ipython.history_manager.get_tail(n=1, include_latest=True)[0][-1]
            return [l.decode() for l in lines] + [final_cell]

    except (ImportError, AttributeError):
        pass

    return ['No logging available. Logging is only available inside Jupyter.']
