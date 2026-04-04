#!/bin/bash
kubectl kustomize k8s/overlays/dev/ > /dev/null
kubectl kustomize k8s/overlays/production/ > /dev/null
echo "Kustomize check complete"
