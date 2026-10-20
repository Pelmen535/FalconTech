"""Select the distinct closed-list profile in one explicit worker process."""
PROFILE='native-tail-closed-v1'
MANIFEST='var/native-closed-tail-20261019/runtime-selection.json'
def configure():
 from . import native_tail as core
 if core._STATES and (core.PROFILE!=PROFILE or core.MANIFEST!=MANIFEST):raise ValueError('Restart worker to switch native-tail profile')
 core.PROFILE=PROFILE;core.MANIFEST=MANIFEST
