lang en_GB.UTF-8
keyboard us
timezone Asia/Shanghai --isUtc

#selinux --enforcing
selinux --disabled

#firewall --enabled --service=cockpit
firewall --disabled

#xconfig --startxonboot
part / --size 8192 --fstype ext4
services --enabled=NetworkManager,sshd --disabled=network


# Root password
auth --useshadow --enablemd5
rootpw --plaintext Kalaisadog2021

repo --name=base --baseurl=http://mirror.centos.org/centos/7/os/x86_64/
repo --name=updates --baseurl=http://mirror.centos.org/centos/7/updates/x86_64/
repo --name=extras --baseurl=http://mirror.centos.org/centos/7/extras/x86_64/
repo --name=epel --baseurl=http://dl.fedoraproject.org/pub/epel/7/x86_64/

%packages 
@core
kernel
dracut
bash
firewalld
NetworkManager
e2fsprogs
rootfiles
docker
openssh-server

#By zhang ranrui
unzip
net-tools
binutils
wget
bash-completion
bc
dmidecode
dmraid
dmraid-events
lvm2
lvm2-libs
kpartx
mdadm
parted
xfsdump
xfsprogs
gdisk
bzip2
extundelete
libHX
libHX-devel
autoconf
gcc
gcc-c++
make
screen
telnet


%end

%post

systemctl enable docker

# By Zhang Ranrui, Add your custom script
wget http://www.rendoumi.com/soft/other/xfs_irecover -O /usr/local/bin/xfs_irecover
chmod 755 /usr/local/bin/xfs_irecover

echo "Banner /etc/issue" >> /etc/ssh/sshd_config

sed -i "s/After=network\.target/After=network-online\.target\nWants=network-online\.target/g" /usr/lib/systemd/system/rc-local.service

chmod 755 /etc/systemd/system/rc.local.service.d
chmod 644 /etc/systemd/system/rc.local.service.d/local.conf

chmod 755 /etc/rc.d/rc.local
systemctl enable rc-local
systemctl start rc-local

# FIXME: it'd be better to get this installed from a package
cat > /etc/rc.d/init.d/livesys << EOF
#!/bin/bash
#
# live: Init script for live image
#
# chkconfig: 345 00 99
# description: Init script for live image.
### BEGIN INIT INFO
# X-Start-Before: display-manager
### END INIT INFO

. /etc/init.d/functions

if ! strstr "\`cat /proc/cmdline\`" rd.live.image || [ "\$1" != "start" ]; then
    exit 0
fi

if [ -e /.liveimg-configured ] ; then
    configdone=1
fi

exists() {
    which \$1 >/dev/null 2>&1 || return
    \$*
}

# Make sure we don't mangle the hardware clock on shutdown
ln -sf /dev/null /etc/systemd/system/hwclock-save.service

livedir="LiveOS"
for arg in \`cat /proc/cmdline\` ; do
  if [ "\${arg##rd.live.dir=}" != "\${arg}" ]; then
    livedir=\${arg##rd.live.dir=}
    return
  fi
  if [ "\${arg##live_dir=}" != "\${arg}" ]; then
    livedir=\${arg##live_dir=}
    return
  fi
done

# enable swaps unless requested otherwise
swaps=\`blkid -t TYPE=swap -o device\`
if ! strstr "\`cat /proc/cmdline\`" noswap && [ -n "\$swaps" ] ; then
  for s in \$swaps ; do
    action "Enabling swap partition \$s" swapon \$s
  done
fi
if ! strstr "\`cat /proc/cmdline\`" noswap && [ -f /run/initramfs/live/\${livedir}/swap.img ] ; then
  action "Enabling swap file" swapon /run/initramfs/live/\${livedir}/swap.img
fi

mountDockerDisk() {
  # support label/uuid
  if [ "\${dockerdev##LABEL=}" != "\${dockerdev}" -o "\${dockerdev##UUID=}" != "\${dockerdev}" ]; then
    dockerdev=\`/sbin/blkid -o device -t "\$dockerdev"\`
  fi

  # if we're given a file rather than a blockdev, loopback it
  if [ "\${dockerdev##mtd}" != "\${dockerdev}" ]; then
    # mtd devs don't have a block device but get magic-mounted with -t jffs2
    mountopts="-t jffs2"
  elif [ ! -b "\$dockerdev" ]; then
    loopdev=\`losetup -f\`
    if [ "\${dockerdev##/run/initramfs/live}" != "\${dockerdev}" ]; then
      action "Remounting live store r/w" mount -o remount,rw /run/initramfs/live
    fi
    losetup \$loopdev \$dockerdev
    dockerdev=\$loopdev
  fi

  # if it's encrypted, we need to unlock it
  if [ "\$(/sbin/blkid -s TYPE -o value \$dockerdev 2>/dev/null)" = "crypto_LUKS" ]; then
    echo
    echo "Setting up encrypted Docker device"
    plymouth ask-for-password --command="cryptsetup luksOpen \$dockerdev EncDocker"
    dockerdev=/dev/mapper/EncDocker
  fi

  # and finally do the mount
  mount \$mountopts \$dockerdev /var/lib/docker
  # if we have /home under what's passed for persistent home, then
  # we should make that the real /home.  useful for mtd device on olpc
  if [ -d /var/lib/docker/docker ]; then mount --bind /var/lib/docker/docker /var/lib/docker ; fi
  [ -x /sbin/restorecon ] && /sbin/restorecon /var/lib/docker
}

findDockerDisk() {
  for arg in \`cat /proc/cmdline\` ; do
    if [ "\${arg##dockerdisk=}" != "\${arg}" ]; then
      dockerdev=\${arg##dockerdisk=}
      return
    fi
  done
}

if strstr "\`cat /proc/cmdline\`" dockerdisk= ; then
  findDockerDisk
elif [ -e /run/initramfs/live/\${livedir}/docker.img ]; then
  dockerdev=/run/initramfs/live/\${livedir}/docker.img
fi

# if we have a persistent /home, then we want to go ahead and mount it
if ! strstr "\`cat /proc/cmdline\`" nodockerdisk && [ -n "\$dockerdev" ] ; then
  action "Mounting persistent /var/lib/docker" mountDockerDisk
fi

# make it so that we don't do writing to the overlay for things which
# are just tmpdirs/caches
mount -t tmpfs -o mode=0755 varcacheyum /var/cache/yum
mount -t tmpfs vartmp /var/tmp
[ -x /sbin/restorecon ] && /sbin/restorecon /var/cache/yum /var/tmp >/dev/null 2>&1

if [ -n "\$configdone" ]; then
  exit 0
fi

# add fedora user with no passwd
action "Adding live user" useradd \$USERADDARGS -c "Live System User" liveuser
passwd -d liveuser > /dev/null
usermod -aG wheel,docker liveuser > /dev/null

# Remove root password lock
passwd -d root > /dev/null2
(echo Kalaisadog2021; echo Kalaisadog2021)|passwd root --stdin

# turn off firstboot for livecd boots
systemctl --no-reload disable firstboot-text.service 2> /dev/null || :
systemctl --no-reload disable firstboot-graphical.service 2> /dev/null || :
systemctl stop firstboot-text.service 2> /dev/null || :
systemctl stop firstboot-graphical.service 2> /dev/null || :

# don't use prelink on a running live image
sed -i 's/PRELINKING=yes/PRELINKING=no/' /etc/sysconfig/prelink &>/dev/null || :

# turn off mdmonitor by default
systemctl --no-reload disable mdmonitor.service 2> /dev/null || :
systemctl --no-reload disable mdmonitor-takeover.service 2> /dev/null || :
systemctl stop mdmonitor.service 2> /dev/null || :
systemctl stop mdmonitor-takeover.service 2> /dev/null || :

# don't enable the gnome-settings-daemon packagekit plugin
gsettings set org.gnome.settings-daemon.plugins.updates active 'false' || :

# don't start cron/at as they tend to spawn things which are
# disk intensive that are painful on a live image
systemctl --no-reload disable crond.service 2> /dev/null || :
systemctl --no-reload disable atd.service 2> /dev/null || :
systemctl stop crond.service 2> /dev/null || :
systemctl stop atd.service 2> /dev/null || :

# Mark things as configured
touch /.liveimg-configured

# add static hostname to work around xauth bug
# https://bugzilla.redhat.com/show_bug.cgi?id=679486
echo "localhost" > /etc/hostname

# Fixing the lang install issue when other lang than English is selected . See http://bugs.centos.org/view.php?id=7217
/usr/bin/cp /usr/lib/python2.7/site-packages/blivet/size.py /usr/lib/python2.7/site-packages/blivet/size.py.orig
/usr/bin/sed -i "s#return self.humanReadable()#return self.humanReadable().encode('utf-8')#g" /usr/lib/python2.7/site-packages/blivet/size.py

EOF

# bah, hal starts way too late
cat > /etc/rc.d/init.d/livesys-late << EOF
#!/bin/bash
#
# live: Late init script for live image
#
# chkconfig: 345 99 01
# description: Late init script for live image.

. /etc/init.d/functions

if ! strstr "\`cat /proc/cmdline\`" rd.live.image || [ "\$1" != "start" ] || [ -e /.liveimg-late-configured ] ; then
    exit 0
fi

exists() {
    which \$1 >/dev/null 2>&1 || return
    \$*
}

touch /.liveimg-late-configured

# read some variables out of /proc/cmdline
for o in \`cat /proc/cmdline\` ; do
    case \$o in
    ks=*)
        ks="--kickstart=\${o#ks=}"
        ;;
    xdriver=*)
        xdriver="\${o#xdriver=}"
        ;;
    esac
done

# if liveinst or textinst is given, start anaconda
if strstr "\`cat /proc/cmdline\`" liveinst ; then
   plymouth --quit
   /usr/sbin/liveinst \$ks
fi
if strstr "\`cat /proc/cmdline\`" textinst ; then
   plymouth --quit
   /usr/sbin/liveinst --text \$ks
fi

# configure X, allowing user to override xdriver
if [ -n "\$xdriver" ]; then
   cat > /etc/X11/xorg.conf.d/00-xdriver.conf <<FOE
Section "Device"
        Identifier      "Videocard0"
        Driver  "\$xdriver"
EndSection
FOE
fi

EOF

chmod 755 /etc/rc.d/init.d/livesys
/sbin/restorecon /etc/rc.d/init.d/livesys
/sbin/chkconfig --add livesys

chmod 755 /etc/rc.d/init.d/livesys-late
/sbin/restorecon /etc/rc.d/init.d/livesys-late
/sbin/chkconfig --add livesys-late

# enable tmpfs for /tmp
systemctl enable tmp.mount


# enable docker
systemctl enable docker.service

# work around for poor key import UI in PackageKit
rm -f /var/lib/rpm/__db*
releasever=$(rpm -q --qf '%{version}\n' --whatprovides system-release)
basearch=$(uname -i)
rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY-fedora-$releasever-$basearch
echo "Packages within this LiveCD"
rpm -qa
# Note that running rpm recreates the rpm db files which aren't needed or wanted
rm -f /var/lib/rpm/__db*

# go ahead and pre-make the man -k cache (#455968)
/usr/bin/mandb

# save a little bit of space at least...
rm -f /boot/initramfs*
# make sure there aren't core files lying around
rm -f /core*

# convince readahead not to collect
# FIXME: for systemd

cat >> /etc/rc.d/init.d/livesys << EOF


# disable updates plugin
cat >> /usr/share/glib-2.0/schemas/org.gnome.settings-daemon.plugins.updates.gschema.override << FOE
[org.gnome.settings-daemon.plugins.updates]
active=false
FOE

# Show the system-config-keyboard tool on the desktop
mkdir /home/liveuser/Desktop -p >/dev/null
cat /usr/share/applications/system-config-keyboard.desktop | sed '/NotShowIn/d' |sed 's/Terminal=false/Terminal=true/' > /home/liveuser/Desktop/system-config-keyboard.desktop
cat /usr/share/applications/liveinst.desktop | sed '/NoDisplay/d' > /home/liveuser/Desktop/liveinst.desktop 
chmod +x /home/liveuser/Desktop/*.desktop
chown -R liveuser:liveuser /home/liveuser

# Liveuser face
if [ -e /usr/share/icons/hicolor/96x96/apps/fedora-logo-icon.png ] ; then
    cp /usr/share/icons/hicolor/96x96/apps/fedora-logo-icon.png /home/liveuser/.face
    chown liveuser:liveuser /home/liveuser/.face
fi

# make the installer show up
if [ -f /usr/share/applications/liveinst.desktop ]; then
  # Show harddisk install in shell dash
  sed -i -e 's/NoDisplay=true/NoDisplay=false/' /usr/share/applications/liveinst.desktop 
  # need to move it to anaconda.desktop to make shell happy
  #cp /usr/share/applications/liveinst.desktop /usr/share/applications/anaconda.desktop
fi
  cat >> /usr/share/glib-2.0/schemas/org.gnome.shell.gschema.override << FOE
[org.gnome.shell]
favorite-apps=['liveinst.desktop','firefox.desktop', 'evolution.desktop', 'empathy.desktop', 'rhythmbox.desktop', 'shotwell.desktop', 'libreoffice-writer.desktop', 'nautilus.desktop', 'gnome-documents.desktop', 'anaconda.desktop']
FOE


# set up auto-login
cat > /etc/gdm/custom.conf << FOE
[daemon]
AutomaticLoginEnable=True
AutomaticLogin=liveuser
FOE

# Turn off PackageKit-command-not-found while uninstalled
if [ -f /etc/PackageKit/CommandNotFound.conf ]; then
  sed -i -e 's/^SoftwareSourceSearch=true/SoftwareSourceSearch=false/' /etc/PackageKit/CommandNotFound.conf
fi

# make sure to set the right permissions and selinux contexts
chown -R liveuser:liveuser /home/liveuser/
restorecon -R /home/liveuser/

# Fixing default locale to us
localectl set-keymap us
localectl set-x11-keymap us
EOF


# rebuild schema cache with any overrides we installed
glib-compile-schemas /usr/share/glib-2.0/schemas


%end
