#!/bin/bash

function traverse() {
  for file in "$1"/*
  do
    if [ ! -d "${file}" ] ; then
      continue
    else
      if [[ "${file}" == *"pycache"* ]] ; then
        rm -rf "${file}"
      else
        traverse "${file}"
      fi
    fi
  done
}

function main() {
  traverse "$1"
}

main "."

