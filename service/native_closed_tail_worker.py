"""Explicit closed-list native worker; reuse established stdout isolation."""
from vreid.native_closed_tail import configure
configure()
from service.native_tail_worker import main
if __name__=='__main__':main()
