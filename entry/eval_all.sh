#!/bin/bash
for f in `cat validation/RECORDS`; do ./next.sh $f; done
