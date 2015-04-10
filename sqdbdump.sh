#!/bin/bash

if [ "$1" == "" ]; then
	echo "USAGE: sqdbdump.sh FILENAME" >&2 
	exit 64
fi

if [ "$SQDBUTIL" == "" ]; then
	export SQDBUTIL="`dirname '$0'`/sqdbutil"
fi

"$SQDBUTIL" "$1" find 0 | while read I; do
	echo "==> $I <=="
	 "$SQDBUTIL" "$1" get $I | hexdump -C
done
