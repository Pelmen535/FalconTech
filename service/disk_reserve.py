"""Fail-closed free-space checks on actual write targets, not runtime location."""
from pathlib import Path
import errno,math,os,shutil,stat

GIB=1024**3

class DiskReserveError(OSError):
    def __init__(self,snapshot):
        self.snapshot=snapshot
        super().__init__(errno.ENOSPC,'Write filesystem unavailable or below reserve: '+'; '.join(snapshot['reasons']))

class WriteDiskReserve:
    def __init__(self,paths,minimum_gib):
        if isinstance(minimum_gib,bool) or not isinstance(minimum_gib,(float,int)) or not math.isfinite(minimum_gib) or minimum_gib<0:
            raise ValueError('Disk reserve must be finite and nonnegative')
        self.minimum_gib=float(minimum_gib);self.paths={}
        self.add(paths)
    def add(self,paths):
        for label,path in paths.items():
            if not isinstance(label,str) or not label or path is None:raise ValueError('Named write path required')
            self.paths[label]=Path(path).expanduser().absolute()
    @staticmethod
    def _existing_directory(path):
        current=path.resolve(strict=False)
        while True:
            try:
                info=current.stat()
                if not stat.S_ISDIR(info.st_mode):raise NotADirectoryError(str(current))
                if not os.access(current,os.W_OK):raise PermissionError(str(current))
                return current,info.st_dev
            except FileNotFoundError:
                parent=current.parent
                if parent==current:raise
                current=parent
    def snapshot(self):
        rows=[];reasons=[];volumes={}
        for label,path in sorted(self.paths.items()):
            row={'role':label,'requested_path':str(path)}
            try:
                existing,device=self._existing_directory(path);usage=shutil.disk_usage(existing)
                row.update(measured_path=str(existing),filesystem_id=int(device),free_bytes=int(usage.free),total_bytes=int(usage.total),minimum_free_bytes=int(self.minimum_gib*GIB))
                key=str(device);volume=volumes.setdefault(key,{'filesystem_id':int(device),'free_bytes':int(usage.free),'roles':[]})
                volume['free_bytes']=min(volume['free_bytes'],int(usage.free));volume['roles'].append(label)
                if usage.free<=0 or usage.free<self.minimum_gib*GIB:
                    reasons.append(label+' filesystem below '+str(self.minimum_gib)+'GiB reserve')
            except (OSError,ValueError) as exc:
                row.update(error_type=type(exc).__name__,available=False)
                reasons.append(label+' filesystem unavailable')
            rows.append(row)
        return {'paths':rows,'filesystems':list(volumes.values()),'reasons':reasons,'minimum_free_GiB':self.minimum_gib,
                'scope':'Each requested directory or its existing ancestor; never sum free bytes across filesystems.'}
    def require(self):
        value=self.snapshot()
        if value['reasons']:raise DiskReserveError(value)
        return value

def manifest_reserve_gib(data):
    import json
    return json.loads(data.decode('utf-8-sig'))['operations']['start_C_disk_free_GiB']
