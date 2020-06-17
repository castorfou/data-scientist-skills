#!/bin/bash
echo "Enter executable to check: "
read executable
path=`which $executable`
echo "Location: $path"
  