#!/bin/sh
vm_id=9999
image_name=CentOS-7-x86_64-GenericCloud.qcow2

qm create ${vm_id} --memory 8196 --net0 virtio,bridge=vmbr0
qm importdisk ${vm_id} ${image_name} local-lvm
qm set ${vm_id} --scsihw virtio-scsi-pci --scsi0 local-lvm:vm-${vm_id}-disk-0
qm set ${vm_id} --ide2 local-lvm:cloudinit
qm set ${vm_id} --boot c --bootdisk scsi0
qm set ${vm_id} --serial0 socket --vga serial0
qm template ${vm_id}
