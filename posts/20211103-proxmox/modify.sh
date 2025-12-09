#!/bin/sh
image_name=CentOS-7-x86_64-GenericCloud.qcow2
# virt-edit -a ${image_name} /etc/cloud/cloud.cfg
virt-edit -a ${image_name} /etc/cloud/cloud.cfg -e 's/disable_root: [Tt]rue/disable_root: False/'
virt-edit -a ${image_name} /etc/cloud/cloud.cfg -e 's/disable_root: 1/disable_root: 0/' 
virt-edit -a ${image_name} /etc/cloud/cloud.cfg -e 's/lock_passwd: [Tt]rue/lock_passwd: False/'
virt-edit -a ${image_name} /etc/cloud/cloud.cfg -e 's/lock_passwd: 1/lock_passwd: 0/' 
virt-edit -a ${image_name} /etc/cloud/cloud.cfg -e 's/ssh_pwauth:   0/ssh_pwauth:   1/'
virt-edit -a ${image_name} /etc/ssh/sshd_config -e 's/PasswordAuthentication no/PasswordAuthentication yes/'
virt-edit -a ${image_name} /etc/ssh/sshd_config -e 's/PermitRootLogin [Nn]o/PermitRootLogin yes/'
virt-edit -a ${image_name} /etc/ssh/sshd_config -e 's/#PermitRootLogin [Yy]es/PermitRootLogin yes/'
virt-edit -a ${image_name} /etc/ssh/sshd_config -e 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/'
virt-edit -a ${image_name} /etc/ssh/sshd_config -e 's/[#M]axAuthTries 6/MaxAuthTries 20/'
virt-customize --install cloud-init,atop,htop,nano,vim,qemu-guest-agent,curl,wget,telnet,lsof,screen -a ${image_name}
