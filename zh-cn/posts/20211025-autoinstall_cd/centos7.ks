text
skipx
install

auth  --useshadow  --enablemd5
authconfig --enableshadow --passalgo=sha512

firstboot --disable
keyboard us
lang en_US.UTF-8
reboot
cdrom

firewall --disable
selinux  --disabled

services --enabled="chronyd"
logging level=info


#ignoredisk --only-use=vda
ignoredisk --only-use=sda
#bootloader --location=mbr --append="net.ifnames=0 biosdevname=0 crashkernel=auto"
bootloader --location=mbr --append="crashkernel=auto"

rootpw --plaintext Renren2021!
timezone Asia/Shanghai --isUtc

network  --device=lo --hostname=localhost.localdomain
user --name=supdev --gid=511 --groups="supdev" --uid=511 --password="Renren2021!"

zerombr
clearpart --all --initlabel 

part biosboot --fstype=biosboot --size=1
part /boot --fstype ext4 --size=2048 
part swap  --asprimary   --size=8192
part /     --fstype ext4 --size=1 --grow

#part biosboot --fstype=biosboot --size=1
#part /boot --fstype ext2 --size 250
#part pv.01 --size 1 --grow
#volgroup vg pv.01
#logvol / --vgname=vg --size=1 --grow --fstype ext4 --fsoptions=discard,noatime --name=root
#logvol /tmp --vgname=vg --size=1024 --fstype ext4 --fsoptions=discard,noatime --name=tmp
#logvol swap --vgname=vg --recommended --name=swap

#uefi
#partition /boot/efi --asprimary --fstype=vfat --label EFI  --size=200
#partition /boot     --asprimary --fstype=ext4 --label BOOT --size=500
#partition /         --asprimary --fstype=ext4 --label ROOT --size=4096 --grow


services --enabled=network

reboot

%pre
parted -s /dev/sda mklabel gpt
%end

%packages
@core
@system-admin-tools
@additional-devel
@virtualization-client
@virtualization-platform
@virtualization-tools
libguestfs-tools-c
perl-Sys-Virt
qemu-guest-agent
qemu-kvm-tools
curl
dstat
expect
openssl
initscripts
ipmitool
lrzsz
lsof
mtools
nc
nmap
perl
perl-CPAN
procps
python
screen
sysstat
systemtap
systemtap-client
systemtap-devel
tcpdump
telnet
vim
wget
wsmancli
zip
chrony
kexec-tools
net-tools
ntp
ntpdate
man
acpid
chrony
telnet
%end
