#!/bin/bash
rm -rf /tmp/bootiso /tmp/bootcustom /tmp/boot.iso
mkdir /tmp/bootiso 
mount -o loop CentOS-7-x86_64-Minimal-2009.iso /tmp/bootiso

mkdir /tmp/bootcustom
cp -r /tmp/bootiso/* /tmp/bootcustom
umount /tmp/bootiso 
rmdir /tmp/bootiso


chmod -R u+w /tmp/bootcustom

cp centos7.ks /tmp/bootcustom/isolinux/ks.cfg

sed -i '/menu\ default/d' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i 's/^timeout\ .*/timeout 10/g' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i '/^label\ linux/i label\ kickstart' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i '/^label\ linux/i \ \ menu\ label\ ^Install\ Using\ Kickstart\ CentOS 7' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i '/^label\ linux/i \ \ menu\ default' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i '/^label\ linux/i \ \ kernel\ vmlinuz\ biosdevname=0' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i '/^label\ linux/i \ \ append\ initrd=initrd.img\ ks=cdrom:\/ks.cfg' /tmp/bootcustom/isolinux/isolinux.cfg
sed -i '/^label\ linux/i \\n' /tmp/bootcustom/isolinux/isolinux.cfg

cd /tmp/bootcustom
mkisofs -o /tmp/boot.iso -b isolinux.bin -c boot.cat -no-emul-boot -boot-load-size 4 -boot-info-table -V "CentOS 7 x86_64" -R -J -v -T isolinux/. .
