"""Keep native TensorRT stdout diagnostics outside the line-delimited JSON RPC."""
import os,sys

def isolate_native_stdout():
    # Preserve a dedicated descriptor for RPC before redirecting C/C++ fd 1.
    control=os.fdopen(os.dup(sys.stdout.fileno()),'w',encoding='utf-8',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    sys.stdout=control
    return control

def main():
    control=isolate_native_stdout()
    try:
        from .vric_road_worker import main as run
        run()
    finally:
        control.close()

if __name__=='__main__':main()
