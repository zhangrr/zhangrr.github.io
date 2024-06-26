#!/usr/bin/python
# $Id: megaclisas-status,v 1.80 2019/08/19 21:20:58 root Exp root $
#
# Written by Adam Cecile <gandalf@NOSPAM.le-vert.net>
# Modified by Vincent S. Cojot <vincent@NOSPAM.cojot.name>
#

import os
import re
import sys
import pdb
import inspect
if sys.platform == 'win32':
	import ctypes

def_megaclipath = "/opt/MegaRAID/MegaCli/MegaCli64"

# Non-Nagios Mode defaults
nagiosmode = False
nagiosoutput=''
nagiosgoodarray = 0
nagiosbadarray = 0
nagiosgooddisk = 0
nagiosbaddisk = 0

# Sane defaults
printarray = True
printcontroller = True
debugmode = False
notempmode = False
totaldrivenumber = 0
totalconfdrivenumber = 0
totalunconfdrivenumber = 0

# Hardcode a max of 16 HBA and 128 LDs for now. LDTable must be initialized to accept populating list of LD's into each ctlr's list.
MaxNumHBA = 16
MaxNumLD = 128
LDTable = [ [] * MaxNumHBA for i in range(MaxNumLD) ]
NestedLDTable = [[False for i in range(MaxNumLD)] for j in range(MaxNumHBA)]

# Outputs is a 'dict' of all MegaCLI outputs so we can re-use them during loops..
Outputs = {}
ConfDisks = {}
NagiosBadDisks = {}
NagiosGoodDisks = {}

# Startup
def print_usage():
	print ('Usage: megaraid-status [--nagios|--debug|--notemp]')

# We need root access to query
if __name__ == '__main__':
	try:
		root_or_admin = os.geteuid() == 0
	except AttributeError:
		root_or_admin = ctypes.windll.shell32.IsUserAnAdmin() !=0
	if not root_or_admin:
		print ('# This script requires Administrator privileges')
		sys.exit(5)

# Check command line arguments to enable nagios or not
if len(sys.argv) > 2:
	print_usage()
	sys.exit(1)

if len(sys.argv) > 1:
	if sys.argv[1] == '--nagios':
		nagiosmode = True
	elif sys.argv[1] == '--debug':
		debugmode = True
	elif sys.argv[1] == '--notemp':
		notempmode = True
	else:
		print_usage()
		sys.exit(1)
# Functions
def dbgprint(msg):
	if (debugmode):
		sys.stderr.write ( str('# DEBUG ('+str(inspect.currentframe().f_back.f_lineno)+') : '+msg+'\n'))

def is_exe(fpath):
	return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def which(program):
	import os
	fpath, fname = os.path.split(program)
	if fpath:
		if is_exe(program):
			return program
	else:
		# Add some defaults
		os.environ["PATH"] += os.pathsep + '/opt/MegaRAID/MegaCli'
		os.environ["PATH"] += os.pathsep + '/ms/dist/hwmgmt/bin'
		os.environ["PATH"] += os.pathsep + '/opt/MegaRAID/perccli'
		os.environ["PATH"] += os.pathsep + '/opt/MegaRAID/storcli'
		os.environ["PATH"] += os.pathsep + '/opt/lsi/storcli'
		os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.realpath(sys.argv[0]))
		for path in os.environ["PATH"].split(os.pathsep):
			dbgprint ('Looking in PATH '+str(path))
			path = path.strip('"')
			exe_file = os.path.join(path, program)
			if is_exe(exe_file):
				dbgprint ('Found "'+program+'" at '+exe_file)
				return exe_file
	return None

# Find MegaCli
for megabin in "MegaCli64","MegaCli","megacli", "MegaCli.exe", "perccli64", "perccli", "storcli64", "storcli":
	dbgprint ('Looking for '+str(megabin)+' in PATH...')
	megaclipath = which(megabin)
	if (megaclipath != None):
		dbgprint ('Will use this executable: '+str(megaclipath))
		break
	
# Check binary exists (and +x), if not print an error message
if (megaclipath != None):
	if os.path.exists(megaclipath) and os.access(megaclipath, os.X_OK):
		pass
	else:
		if nagiosmode:
			print ('UNKNOWN - Cannot find '+megaclipath)
		else:
			print ('Cannot find ' + megaclipath + 'in your PATH. Please install it.')
		sys.exit(3)
else:
	print ('Cannot find "MegaCli{64,}", "megacli{64,}", "perccli{64,}" or "storcli{64,}" in your PATH. Please install one of them.')
	sys.exit(3)


#### pdb.set_trace()

def returnWdthFromArrayCol(glarray,idx):
	maxwdth = 0
	for glrow in glarray:
		if ( len(glrow[idx]) > maxwdth):
			maxwdth = len(glrow[idx])
	return maxwdth

# Get and cache command output
def getOutput(cmd):
	lines = []
	if cmd in Outputs:
		dbgprint ("Got Cached value: "+str(cmd))
		lines = Outputs[cmd]
	else:
		dbgprint ("Not a Cached value: "+str(cmd))
		output = os.popen(cmd)
		for line in output:
			if not re.match(r'^$',line.strip()):
				lines.append(line.strip())
		Outputs[cmd] = lines
	return lines
 
# Get and cache disks, make sure we don't count the same disk twice
def AddDisk(mytable,disk):
        lines = []
        if disk in mytable:
                dbgprint ("Disk: "+str(disk)+" Already present in Disk Table")
                return False
        else:
                dbgprint ("Confed "+str(nagiosgooddisk)+'/'+str(nagiosbaddisk)+"Disk: "+str(disk)+" Not already present in Disk Table, adding")
                mytable[disk] = True
                return True

def returnControllerNumber(output):
	for line in output:
		if re.match(r'^Controller Count.*$',line.strip()):
			return int(line.split(':')[1].strip().strip('.'))

def returnTotalDriveNumber(output):
	for line in output:
		if re.match(r'Number of Physical Drives on Adapter.*$',line.strip()):
			return int(line.split(':')[1].strip())

def returnRebuildProgress(output):
	percent = 0
	tmpstr = ''
	for line in output:
		if re.match(r'^Rebuild Progress on Device at Enclosure.*, Slot .* Completed ',line.strip()):
			tmpstr = line.split('Completed')[1].strip()
			percent = int(tmpstr.split('%')[0].strip())
	return percent

def returnConfDriveNumber(controllerid,output):
	# Count the configured drives
	confdrives = 0 ; enclid = 'N/A' ; slotid = 'N/A'
	for line in output:

		if re.match(r'Enclosure Device ID: .*$',line.strip()):
			# We match here early in the analysis so reset the vars if this is a new disk we're reading..
			enclid = line.split(':')[1].strip()
		elif re.match(r'Slot Number: .*$',line.strip()):
			slotid = line.split(':')[1].strip()
			if ( AddDisk(ConfDisks, str(controllerid) + enclid + slotid)):
				confdrives += 1
	return int(confdrives)

def returnUnConfDriveNumber(output):
	# Count the un-configured/Hotspare drives
	unconfdrives = 0
	for line in output:
		if re.match(r'^Firmware state: Unconfigured.*$',line.strip()):
			unconfdrives += 1
		elif re.match(r'^Firmware state: Hotspare.*$',line.strip()):
			unconfdrives += 1
	return int(unconfdrives)

def returnControllerModel(output):
	for line in output:
		if re.match(r'^Product Name.*$',line.strip()):
			return line.split(':')[1].strip()

def returnMemorySize(output):
	for line in output:
		if re.match(r'^Memory Size.*$',line.strip()):
			return line.split(':')[1].strip()

def returnFirmwareVersion(output):
	for line in output:
		if re.match(r'^FW Package Build.*$',line.strip()):
			return line.split(':')[1].strip()

def returnROCTemp(output):
	ROCtemp = ''
	tmpstr = ''
	if (notempmode):
		return str('N/A')
	else:
		for line in output:
			if re.match(r'^ROC temperature :.*$',line.strip()):
				tmpstr = line.split(':')[1].strip()
				ROCtemp = re.sub(' +.*$', '', tmpstr)
		if ( ROCtemp != '' ):
			return str(str(ROCtemp)+'C')
		else:
			return str('N/A')

def returnBBUPresence(output):
	BBU = ''
	tmpstr = ''
	for line in output:
		if re.match(r'^BBU +:.*$',line.strip()):
			tmpstr = line.split(':')[1].strip()
			BBU = re.sub(' +.*$', '', tmpstr)
			break
	if ( BBU != '' ):
		return str(BBU)
	else:
		return str('N/A')

def returnBBUStatus(output):
	BBUStatus = ''
	tmpstr = ''
	for line in output:
		if re.match(r'^ *Battery Replacement required +:.*$',line.strip()):
			tmpstr = line.split(':')[1].strip()
			BBUStatus = re.sub(' +.*$', '', tmpstr)
			break
	if ( BBUStatus == 'Yes' ):
		return str('REPL')
	else:
		return str('Good')

def returnArrayNumber(output):
	i = 0
	for line in output:
		if re.match(r'^(CacheCade )?Virtual Drive:.*$',line.strip()):
			i += 1
	return i

def returnHBAPCIInfo(output):
	busprefix = '0000'
	busid = ''
	devid = ''
	functionid = ''
	pcipath = ''
	for line in output:
		if re.match(r'^Bus Number.*:.*$',line.strip()):
			busid = str(line.strip().split(':')[1].strip()).zfill(2)
		if re.match(r'^Device Number.*:.*$',line.strip()):
			devid = str(line.strip().split(':')[1].strip()).zfill(2)
		if re.match(r'^Function Number.*:.*$',line.strip()):
			functionid = str(line.strip().split(':')[1].strip()).zfill(1)
	if busid:
		pcipath = str(busprefix + ':' + busid + ':' + devid + '.' + functionid)
		dbgprint("Array PCI path : "+pcipath)
		return str(pcipath)
	else:
		return None

def returnHBAInfo(table,output,controllerid):
	controllermodel = 'Unknown'
	controllerram = 'Unknown'
	controllerrev = 'Unknown'
	controllertemp = ''
	controllermodel = returnControllerModel(output)
	controllerram = returnMemorySize(output)
	controllerrev = returnFirmwareVersion(output)
	controllertemp = returnROCTemp(output)
	controllerbbu = returnBBUPresence(output)
	if controllerbbu == 'Present':
		cmd = '%s -AdpBbuCmd -GetBbuStatus -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		controllerbbu = returnBBUStatus(output)

	if controllermodel != 'Unknown':
		table.append([ 'c'+str(controllerid), controllermodel, controllerram, str(controllertemp), str(controllerbbu), str('FW: '+controllerrev) ])

def returnArrayInfo(output,controllerid,arrayid,arrayindex):
	id = 'c'+str(controllerid)+'u'+str(arrayid)
	operationlinennumber = False
	linenumber = 0
	targetid = ''
	raidtype = ''
	raidlvl = ''
	size = ''
	state = 'N/A'
	strpsz = ''
	dskcache = 'N/A'
	properties = ''
	spandepth = 0
	diskperspan = 0
	cachecade_info = 'None'

	for line in output:
		if re.match(r'^(CacheCade )?Virtual Drive:.*(Target Id: [0-9]+).*$',line.strip()):
			# Extract the SCSI Target ID
			targetid = line.strip().split(':')[2].split(')')[0].strip()
		elif re.match(r'^RAID Level.*?:.*$',line.strip()):
			# Extract the primary raid type, decide on X0 RAID level later when we hit Span Depth
			raidlvl = int(line.strip().split(':')[1].split(',')[0].split('-')[1].strip())
		elif re.match(r'^Size.*?:.*$',line.strip()):
			# Size reported in MB
			if re.match(r'^.*MB$',line.strip().split(':')[1]):
				size = line.strip().split(':')[1].strip('MB').strip()
				if ( float(size) > 1000):
					size = str(int(round((float(size) / 1000))))+'G'
				else:
					size = str(int(round(float(size))))+'M'
			# Size reported in TB
			elif re.match(r'^.*TB$',line.strip().split(':')[1]):
				size = line.strip().split(':')[1].strip('TB').strip()
				size = str(int(round((float(size) * 1000))))+'G'
			# Size reported in GB (default)
			else:
				size = line.strip().split(':')[1].strip('GB').strip()
				size = str(int(round((float(size)))))+'G'
		elif re.match(r'^Span Depth.*?:.*$',line.strip()):
			# If Span Depth is greater than 1 chances are we have a RAID 10, 50 or 60
			spandepth = line.strip().split(':')[1].strip()
		elif re.match(r'^State.*?:.*$',line.strip()):
			state = line.strip().split(':')[1].strip()
		elif re.match(r'^Strip Size.*?:.*$',line.strip()):
			strpsz = line.strip().split(':')[1].strip()
		elif re.match(r'^Number Of Drives per span.*:.*$',line.strip()):
			diskperspan = int(line.strip().split(':')[1].strip())
		elif re.match(r'^Current Cache Policy.*?:.*$',line.strip()):
			props = line.strip().split(':')[1].strip()
			if re.search('ReadAdaptive', props):
				properties += 'ADRA'
			if re.search('ReadAhead', props):
				properties += 'RA'
			if re.match('ReadAheadNone', props):
				properties += 'NORA'
			if re.search('WriteBack', props):
				properties += ',WB'
			if re.match('WriteThrough', props):
				properties += ',WT'
		elif re.match(r'^Disk Cache Policy.*?:.*$',line.strip()):
			props = line.strip().split(':')[1].strip()
			if re.search('Disabled', props):
				dskcache = 'Disabled'
			if re.search('Disk.s Default', props):
				dskcache = 'Default'
			if re.search('Enabled', props):
				dskcache = 'Enabled'
		elif re.match(r'^Ongoing Progresses.*?:.*$',line.strip()):
			operationlinennumber = linenumber
		elif re.match(r'Cache Cade Type\s*:.*$', line):
			cachecade_info = "Type : " + line.strip().split(':')[1].strip()
		elif re.match(r'^Target Id of the Associated LDs\s*:.*$', line):
			associated=[]
			for array in line.split(':')[1].strip().split(','):
					if array.isdigit():
						associated.append('c%du%d' % (controllerid, int(array)))
			if len(associated) >= 1:
				cachecade_info = "Associated : %s" %(', '.join(associated))
		linenumber += 1

	# If there was an ongoing operation, find the relevant line in the previous output
	if operationlinennumber:
		inprogress = str(output[operationlinennumber + 1])
		# some ugly output fix..
		str1 = inprogress.split(':')[0].strip()
		str2 = inprogress.split(':')[1].strip()
		inprogress = str1+" : "+str2
	else:
		inprogress = 'None'

	# Compute the RAID level
	NestedLDTable[int(controllerid)][int(arrayindex)] = False
	if raidlvl == '':
		raidtype = str('N/A')
	else:
		if (int(spandepth) >= 2):
			raidtype = str('RAID-' + str(raidlvl) + '0')
			NestedLDTable[controllerid][int(arrayindex)] = True
		else:
			if(raidlvl == 1):
				if(diskperspan > 2):
					raidtype = str('RAID-10')
					NestedLDTable[controllerid][int(arrayindex)] = True
				else:
					raidtype = str('RAID-' + str(raidlvl))
			else:
				raidtype = str('RAID-' + str(raidlvl))

	dbgprint('RAID Level: ' + str(raidlvl)
		+ ' Span Depth: ' + str(spandepth)
		+ ' Disk Per Span: ' + str(diskperspan)
		+ ' Raid Type: ' + str(raidtype))
	return [id,raidtype,size,strpsz,properties,dskcache,state,targetid,cachecade_info,inprogress]

def returnDiskInfo(output,controllerid):
	arrayid = False
	arrayindex = -1
	sarrayid = 'Unknown'
	diskid = False
	oldenclid = False
	enclid = False
	spanid = False
	slotid = False
	lsidid = 'Unknown'
	table = []
	fstate = 'Offline'
	substate = 'Unknown'
	model = 'Unknown'
	speed = 'Unknown'
	dsize = 'Unknown'
	temp = 'Unk0C'
	percent = 0
	for line in output:
		if re.match(r'^Span: [0-9]+ - Number of PDs:',line.strip()):
			spanid = line.split(':')[1].strip()
			spanid = re.sub(' - Number of PDs.*', '', spanid)
		elif re.match(r'Enclosure Device ID: .*$',line.strip()):
			# We match here early in the analysis so reset the vars if this is a new disk we're reading..
			oldenclid = enclid
			enclid = line.split(':')[1].strip().replace("N/A","")
			if oldenclid != False:
				fstate = 'Offline'
				model = 'Unknown'
				speed = 'Unknown'
				temp = 'Unk0C'
				slotid = False
				lsidid = 'Unknown'
		elif re.match(r'^Coerced Size: ',line.strip()):
			dsize = line.split(':')[1].strip()
			dsize = re.sub(' \[.*\.*$', '', dsize)
			dsize = re.sub('[0-9][0-9] GB', ' Gb', dsize)
		elif re.match(r'^(CacheCade )?Virtual (Disk|Drive): [0-9]+.*$',line.strip()):
			arrayindex += 1
			arrayid = line.split('(')[0].split(':')[1].strip()
		elif re.match(r'^Drive.s posi*tion: DiskGroup: [0-9]+,.*$',line.strip()):
			notarrayid = line.split(',')[1].split(':')[1].strip()
		elif re.match(r'PD: [0-9]+ Information.*$',line.strip()):
			diskid = line.split()[1].strip()
		elif re.match(r'^Device Id: .*$',line.strip()):
			lsidid = line.split(':')[1].strip()
		elif re.match(r'Slot Number: .*$',line.strip()):
			slotid = line.split(':')[1].strip()
		elif re.match(r'Firmware state: .*$',line.strip()):
			fstate = line.split(':')[1].strip()
			subfstate = re.sub('\(.*', '', fstate)
			dbgprint('Firmware State: '+str(fstate)+' '+str(subfstate))
		elif re.match(r'Inquiry Data: .*$',line.strip()):
			model = line.split(':')[1].strip()
			model = re.sub(' +', ' ', model)

			# re-define our "sub-code"
			# our seagate drives have an ID string of
			# 'Z1E19S2QST2000DM001-1CH164                      CC43'
			# or
			# '6XW02738ST32000542AS                            CC32'

			m = re.match(r'(\w{8})(ST\w+)(?:-(\w{6}))?(?:\s+(\w+))', model)
			if m:
			    if m.group(3):
				model = '{}-{} {} {}'.format(m.group(2), m.group(3), m.group(4), m.group(1))
			    else:
				model = '{} {:>10} {}'.format(m.group(2), m.group(4), m.group(1))
			    continue

			# Sub code
			manuf = re.sub(' .*', '', model)
			dtype = re.sub(manuf+' ', '', model)
			dtype = re.sub(' .*', '', dtype)
			hwserial = re.sub('.*'+dtype+' *', '', model)
		elif re.match(r'^Media Type: .*$',line.strip()):
			mtype = line.split(':')[1].strip()
			if mtype == 'Hard Disk Device':
				mtype = 'HDD'
			else:
				if mtype == 'Solid State Device':
					mtype = 'SSD'
				else:
					mtype = 'N/A'
		elif re.match(r'Device Speed: .*$',line.strip()):
			speed = line.split(':')[1].strip()
		elif re.match(r'Drive Temperature :.*$',line.strip()):
			if (notempmode):
				temp = 'N/A'
			else:
				# Drive temp is amongst the last few lines matched, decide here if we add information to the table..
				temp = line.split(':')[1].strip()
				temp = re.sub(' \(.*\)', '', temp)
			if model != 'Unknown':
				dbgprint('Disk Info: '+str(arrayid)+' '+str(diskid)+' '+str(oldenclid))
				if subfstate == 'Rebuild':
                                        cmd = '%s pdrbld -showprog -physdrv\[%s:%s\] -a%d -NoLog' % (megaclipath, enclid, slotid, controllerid)
                                        output = getOutput(cmd)
                                        percent = returnRebuildProgress(output)
                                        fstate = str('Rebuilding (%d%%)' % (percent))

				if (( NestedLDTable[controllerid][int(arrayindex)] == True) and (spanid != False)):
					sarrayid = str(arrayid)+"s"+spanid
				else:
					sarrayid = str(arrayid)
				table.append([sarrayid, str(diskid), mtype, model, dsize, fstate , speed, temp, enclid, slotid, lsidid])
	return table


def returnUnconfDiskInfo(output,controllerid):
	arrayid = False
	diskid = False
	olddiskid = False
	enclid = False
	slotid = False
	lsidid = 'Unknown'
	table = []
	fstate = 'Offline'
	substate = 'Unknown'
	model = 'Unknown'
	speed = 'Unknown'
	mtype = 'Unknown'
	dsize = 'Unknown'
	temp = 'Unk0C'
	ospath = 'N/A'
	for line in output:
		if re.match(r'Enclosure Device ID: .*$',line.strip()):
		# We match here early in the analysis so reset the vars if this is a new disk we're reading..
			oldenclid = enclid
			enclid = line.split(':')[1].strip().replace("N/A","")
			if oldenclid != False:
				arrayid = False
				fstate = 'Offline'
				model = 'Unknown'
				speed = 'Unknown'
				temp = 'Unk0C'
				slotid = False
				lsidid = 'Unknown'

		elif re.match(r'^Coerced Size: ',line.strip()):
			dsize = line.split(':')[1].strip()
			dsize = re.sub(' \[.*\.*$', '', dsize)
			dsize = re.sub('[0-9][0-9] GB', ' Gb', dsize)
		elif re.match(r'^Drive.s posi*tion: DiskGroup: [0-9]+,.*$',line.strip()):
			arrayid = line.split(',')[1].split(':')[1].strip()
		elif re.match(r'^Device Id: [0-9]+.*$',line.strip()):
			diskid = line.split(':')[1].strip()
		elif re.match(r'Slot Number: .*$',line.strip()):
			slotid = line.split(':')[1].strip()
		elif re.match(r'Firmware state: .*$',line.strip()):
			fstate = line.split(':')[1].strip()
			subfstate = re.sub('\(.*', '', fstate)
			dbgprint('Firmware State: '+str(fstate)+' '+str(subfstate))
		elif re.match(r'Inquiry Data: .*$',line.strip()):
			model = line.split(':')[1].strip()
			model = re.sub(' +', ' ', model)

			# re-define our "sub-code"
			# our seagate drives have an ID string of
			# 'Z1E19S2QST2000DM001-1CH164                      CC43'
			# or
			# '6XW02738ST32000542AS                            CC32'

			m = re.match(r'(\w{8})(ST\w+)(?:-(\w{6}))?(?:\s+(\w+))', model)
			if m:
			    if m.group(3):
				model = '{}-{} {} {}'.format(m.group(2), m.group(3), m.group(4), m.group(1))
			    else:
				model = '{} {:>10} {}'.format(m.group(2), m.group(4), m.group(1))
			    continue

			manuf = re.sub(' .*', '', model)
			dtype = re.sub(manuf+' ', '', model)
			dtype = re.sub(' .*', '', dtype)
			hwserial = re.sub('.*'+dtype+' *', '', model)
		elif re.match(r'^Media Type: .*$',line.strip()):
			mtype = line.split(':')[1].strip()
			if mtype == 'Hard Disk Device':
				mtype = 'HDD'
			else:
				if mtype == 'Solid State Device':
					mtype = 'SSD'
				else:
					mtype = 'N/A'
		elif re.match(r'Device Speed: .*$',line.strip()):
			speed = line.split(':')[1].strip()
		elif re.match(r'Drive Temperature :.*$',line.strip()):
			# Drive temp is amongst the last few lines matched, decide here if we add information to the table..
			if (notempmode):
				temp = 'N/A'
			else:
				temp = line.split(':')[1].strip()
				temp = re.sub('\(.*\)', '', temp)
			if arrayid == False:
				if subfstate == 'Unconfigured':
					dbgprint('Unconfigured Disk: Arrayid: '+str(arrayid)+' DiskId: '+str(diskid)+' '+str(olddiskid)+' '+str(fstate))
				elif subfstate == 'Online, Spun Up':
					dbgprint('Online Unconfed Disk: Arrayid: '+str(arrayid)+' DiskId: '+str(diskid)+' '+str(olddiskid)+' '+str(fstate))
				table.append([ mtype, model, dsize, fstate, speed, temp, enclid, slotid, diskid, ospath])
	return table

cmd = '%s -adpCount -NoLog' % (megaclipath)
output = getOutput(cmd)
controllernumber = returnControllerNumber(output)

bad = False

# List available controller
if printcontroller:
	if controllernumber:
		if not nagiosmode:
			print ('-- Controller information --')

		i = 0
		controllerid = 0
		mlen = 0
		hbainfo = []
		while controllerid < controllernumber:
			cmd = '%s -AdpAllInfo -a%d -NoLog' % (megaclipath, controllerid)
			output = getOutput(cmd)
			returnHBAInfo(hbainfo, output,controllerid)
			controllerid += 1
		mlen = returnWdthFromArrayCol(hbainfo,1)

		controllerid = 0
		for hba in hbainfo:
			hbafmt = str('%-5s | %-'+str(mlen)+'s | %-6s | %-4s | %-6s | %-12s ')
			# Header
			if ( i == 0 ):
				if not nagiosmode:
					print (hbafmt % ("-- ID","H/W Model","RAM","Temp","BBU", "Firmware"))
			if not nagiosmode:
				print (hbafmt % (
					hba[0],
					hba[1],
					hba[2],
					hba[3],
					hba[4],
					hba[5]))
			i += 1
		if not nagiosmode:
			print ('')
	else:
		print ("No MegaRAID or PERC adapter detected on your system!")
		exit(1)

if printarray:
	if not nagiosmode:
		print ('-- Array information --')

	controllerid = 0
	pcipath = ''
	diskpath = ''
	i = 0 ; j = 0
	mlen = 0 ; rlen = 0 ; clen = 0
	while controllerid < controllernumber:
		arrayindex = 0

		cmd = '%s -LDInfo -lall -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraynumber = returnArrayNumber(output)
		# We need to explore each HBA to look for gaps in LD's
		ldid = 0 ; ldcount = 0
		while ldcount < arraynumber:
			cmd = '%s -LDInfo -l%d -a%d -NoLog' % (megaclipath, ldid, controllerid)
			output = getOutput(cmd)
			for line in output:
				if re.match(r'^Adapter.*Virtual Drive .* Does not Exist',line.strip()):
					ldid += 1
				elif re.match(r'^(CacheCade )?Virtual Drive:',line.strip()):
					LDTable[controllerid].append ( ldid )
					#NestedLDTable[controllerid][int(arrayindex)] = False
					ldcount += 1
					ldid += 1

		while arrayindex < arraynumber:
			ldid = LDTable[controllerid][arrayindex]
			cmd = '%s -LDInfo -l%d -a%d -NoLog' % (megaclipath, ldid, controllerid)
			output = getOutput(cmd)
			arrayinfo = returnArrayInfo(output, controllerid, ldid, arrayindex)
			if ( len(arrayinfo[1]) > rlen):
				rlen = len(arrayinfo[1])
			if ( len(arrayinfo[4]) > mlen):
				mlen = len(arrayinfo[4])
			if ( len(arrayinfo[8]) > clen):
				clen = len(arrayinfo[8])
			arrayindex += 1
		controllerid += 1

	controllerid = 0
	while controllerid < controllernumber:
		arrayindex = 0

		cmd = '%s -AdpGetPciInfo -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		pcipath = returnHBAPCIInfo(output)

		cmd = '%s -LDInfo -lall -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraynumber = returnArrayNumber(output)
		while arrayindex < arraynumber:
			ldid = LDTable[controllerid][arrayindex]
			cmd = '%s -LDInfo -l%d -a%d -NoLog' % (megaclipath, ldid, controllerid)
			output = getOutput(cmd)
			arrayinfo = returnArrayInfo(output,controllerid, ldid, arrayindex)

			if pcipath:
				diskprefix = str('/dev/disk/by-path/pci-' + pcipath + '-scsi-0:')
				dbgprint('Will look for DISKprefix : ' + diskprefix)
				# RAID disks are usually with a channel of '2', JBOD disks with a channel of '0'
				for j in range (1, 8):
					diskpath = diskprefix + str(j) + ':' + str(arrayinfo[7]) + ':0'
					dbgprint('Looking for DISKpath : ' + diskpath)
					if os.path.exists(diskpath):
						arrayinfo[7] = os.path.realpath(diskpath)
						dbgprint('Found DISK match: ' + diskpath + ' -> ' + arrayinfo[7])
						break
			else:
				arrayinfo[7] = 'N/A'

			# Pad the string length, just to make sure it's aligned with the headers...
			if (rlen < len("Type")):
				rlen = len("Type")
			if (mlen < len("Flags")):
				mlen = len("Flags")
			if (clen < len("CacheCade")):
				clen = len("CacheCade")

			ldfmt = str('%-5s | %-'+str(rlen)+'s | %7s | %7s | %'+str(mlen)+'s | %8s | %8s | %8s | %-'+str(clen)+'s |%-12s ')
			# Header
			if ( i == 0 ):
				if not nagiosmode:
					print (ldfmt % ("-- ID", "Type", "Size", "Strpsz", "Flags", "DskCache", "Status", "OS Path", "CacheCade", "InProgress" ))
			if not nagiosmode:
				print (ldfmt % (
					arrayinfo[0],
					arrayinfo[1],
					arrayinfo[2],
					arrayinfo[3],
					arrayinfo[4],
					arrayinfo[5],
					arrayinfo[6],
					arrayinfo[7],
					arrayinfo[8],
					arrayinfo[9]))
			dbgprint('Array state : LD ' + arrayinfo[0] + ', status : ' + arrayinfo[6])
			if arrayinfo[6] not in [ 'Optimal', 'N/A' ]:
				bad = True
				nagiosbadarray += 1
			else:
				nagiosgoodarray += 1
			arrayindex += 1
			i += 1
		controllerid += 1
	if not nagiosmode:
		print ('')

controllerid = 0
while controllerid < controllernumber:
	cmd = '%s -PDGetNum -a%d -NoLog' % (megaclipath, controllerid)
	output = getOutput(cmd)
	totaldrivenumber += returnTotalDriveNumber(output)
	controllerid += 1

if totaldrivenumber:
	if not nagiosmode:
		print ('-- Disk information --')

	i = 0
	dlen = 0 ; mlen = 0 ; flen = 0
	controllerid = 0
	while controllerid < controllernumber:
		arrayid = 0
		cmd = '%s -LDInfo -lall -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraynumber = returnArrayNumber(output)
		#### BUG: -LdPdInfo shows all PD on the adapter, not just for the LD we wanted..
		#### while arrayid <= arraynumber:
		cmd = '%s -LdPdInfo -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraydisk = returnDiskInfo(output,controllerid)
		for array in arraydisk:
			diskname = str(controllerid) + array[8] + array[9]
			dbgprint('Disk c'+diskname + ' status : ' + array[5])
			if re.match("|".join([ '^Online$', '^Online, Spun Up$', '^Rebuilding \(.*' ]), array[5]):
				if ( AddDisk(NagiosGoodDisks, diskname) ):
					nagiosgooddisk += 1
			else:
				bad = True
				if ( AddDisk(NagiosBadDisks, diskname) ):
					nagiosbaddisk += 1

		if ( returnWdthFromArrayCol(arraydisk,0) > dlen):
			dlen = returnWdthFromArrayCol(arraydisk,0)
		if ( returnWdthFromArrayCol(arraydisk,3) > mlen):
			mlen = returnWdthFromArrayCol(arraydisk,3)
		if ( returnWdthFromArrayCol(arraydisk,5) > flen):
			flen = returnWdthFromArrayCol(arraydisk,5)
		controllerid += 1

	controllerid = 0
	while controllerid < controllernumber:
		arrayid = 0

		cmd = '%s -LDInfo -lall -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraynumber = returnArrayNumber(output)
		#### BUG: -LdPdInfo shows all PD on the adapter, not just for said LD..
		#### while arrayid <= arraynumber:

		cmd = '%s -LdPdInfo -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraydisk = returnDiskInfo(output,controllerid)

		# Adjust print format with width computed above
		drvfmt = "%-"+str(dlen+5)+"s | %-4s | %-"+str(mlen)+"s | %-8s | %-"+str(flen)+"s | %-8s | %-4s | %-8s | %-8s"
		for array in arraydisk:
			# Header
			if ( i == 0 ):
				if not nagiosmode:
					print (drvfmt % (
					"-- ID", "Type", "Drive Model", "Size", "Status", "Speed", "Temp", "Slot ID", "LSI ID"))
			# Drive information
			if not nagiosmode:
				print (drvfmt % (
					str('c'+str(controllerid)+'u'+array[0]+'p'+array[1]),		# c0p0
					array[2],							# HDD/SDD
					array[3],							# Model Information (Variable len)
					array[4],							# Size
					array[5],							# Status (Variable len)
					array[6],							# Speed
					array[7],							# Temp
					str('['+array[8]+':'+array[9]+']'),				# Slot ID
					array[10]))							# LSI ID
			i = i + 1
		controllerid += 1
	if not nagiosmode:
		print ('')

controllerid = 0
totalconfdrivenumber = 0
totalunconfdrivenumber = 0
totaldrivenumber = 0
while controllerid < controllernumber:
	cmd = '%s -LdPdInfo -a%d -NoLog' % (megaclipath, controllerid)
	output = getOutput(cmd)
	totalconfdrivenumber += returnConfDriveNumber(controllerid,output)

	cmd = '%s -PDGetNum -a%d -NoLog' % (megaclipath, controllerid)
	output = getOutput(cmd)
	totaldrivenumber += returnTotalDriveNumber(output)

	cmd = '%s -PDList -a%d -NoLog' % (megaclipath, controllerid)
	output = getOutput(cmd)
	# Sometimes a drive will be reconfiguring without any info on that it is going through a rebuild process.
	# This happens when expanding an R{5,6,50,60} array, for example. In that case, totaldrivenumber will still be
	# greater than totalconfdrivenumber while returnUnConfDriveNumber(output) will be zero. The math below attempts to solve this.
	totalunconfdrivenumber += max(returnUnConfDriveNumber(output), totaldrivenumber - totalconfdrivenumber)

	controllerid += 1

dbgprint('Total Drives in system : ' + str(totaldrivenumber))
dbgprint('Total Configured Drives : ' + str(totalconfdrivenumber))
dbgprint('Total Unconfigured Drives : ' + str(totalunconfdrivenumber))

if totalunconfdrivenumber:
	if not nagiosmode:
		print ('-- Unconfigured Disk information --')

	controllerid = 0
	pcipath = ''
	while controllerid < controllernumber:
		arrayid = 0

		cmd = '%s -LDInfo -lall -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraynumber = returnArrayNumber(output)
		cmd = '%s -AdpGetPciInfo -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		pcipath = returnHBAPCIInfo(output)
		#### BUG: -LdPdInfo shows all PD on the adapter, not just for given LD..
		#### while arrayid <= arraynumber:

		cmd = '%s -PDList -a%d -NoLog' % (megaclipath, controllerid)
		output = getOutput(cmd)
		arraydisk = returnUnconfDiskInfo(output,controllerid)
		for array in arraydisk:
			dbgprint('Unconfed '+str(nagiosgooddisk)+'/'+str(nagiosbaddisk)+' Disk c'+str(controllerid)+'uXpY status : ' + array[3])
			if array[3] in [ 'Online', 'Unconfigured(good), Spun Up', 'Unconfigured(good), Spun down', 'JBOD','Hotspare, Spun Up','Hotspare, Spun down','Online, Spun Up' ]:
				nagiosgooddisk += 1
			else:
				bad = True
				nagiosbaddisk += 1

			# JBOD disks has a real device path and are not masked. Try to find a device name here, if possible.
			if pcipath:
				if array[3] in [ 'JBOD' ]:
					diskprefix = str('/dev/disk/by-path/pci-' + pcipath + '-scsi-0:0:')
					dbgprint('Will look for DISKprefix : ' + diskprefix)
					# RAID disks are usually with a channel of '2', JBOD disks with a channel of '0'
					diskpath = diskprefix + str(array[8]) + ':0'
					dbgprint('Looking for DISKpath : ' + diskpath)
					if os.path.exists(diskpath):
						dbgprint('Found DISK match: ' + diskpath + ' -> ' + array[9])
						array[9] = os.path.realpath(diskpath)
					else:
						dbgprint('DISK NOT present: ' + diskpath)
						array[9] = 'N/A'

		mlen = returnWdthFromArrayCol(arraydisk,1)
		flen = returnWdthFromArrayCol(arraydisk,3)

		# Adjust print format with widths computed above
		drvfmt = "%-7s | %-4s | %-"+str(mlen)+"s | %-8s | %-"+str(flen+2)+"s | %-8s | %-4s | %-8s | %-6s | %-8s"
		i = 0
		for array in arraydisk:
			# Header
			if ( i == 0 ):
				if not nagiosmode:
					print (drvfmt % (
					"-- ID", "Type", "Drive Model", "Size", "Status", "Speed", "Temp", "Slot ID", "LSI ID", "Path"))
			# Drive information
			if not nagiosmode:
				print (drvfmt % (
					str('c'+str(controllerid)+'uXpY'),				# cXpY
					array[0],							# HDD/SDD
					array[1],							# Model Information (Variable len)
					array[2],							# Size
					array[3],							# Status (Variable len)
					array[4],							# Speed
					array[5],							# Temp
					str('['+array[6]+':'+array[7]+']'),				# Slot ID
					array[8],							# LSI ID
					array[9]))							# OS path, if any
			i += 1
		controllerid += 1
	if not nagiosmode:
		print ('')

if (debugmode):
	dbgprint ('Printing Outputs[][]')
	for myl in Outputs:
		dbgprint(myl+'\n')
		sys.stderr.write("\n".join("".join(map(str,myd)) for myd in Outputs[myl])+'\n')
	dbgprint ('Printing arraydisk[]')
	sys.stderr.write("\n".join(" | ".join(map(str,myd)) for myd in arraydisk)+'\n')
	dbgprint ('Printing ConfDisks[]')
	sys.stderr.write("\n".join("".join(map(str,myd)) for myd in ConfDisks)+'\n')
	dbgprint ('Printing NagiosGoodDisks[]')
	sys.stderr.write("\n".join("".join(map(str,myd)) for myd in NagiosGoodDisks)+'\n')
	dbgprint ('Printing NagiosBadDisks[]')
	sys.stderr.write("\n".join("".join(map(str,myd)) for myd in NagiosBadDisks)+'\n')

if nagiosmode:
	if bad:
		print ('RAID ERROR - Arrays: OK:'+str(nagiosgoodarray)+' Bad:'+str(nagiosbadarray)+' - Disks: OK:'+str(nagiosgooddisk)+' Bad:'+str(nagiosbaddisk))
		sys.exit(2)
	else:
		print ('RAID OK - Arrays: OK:'+str(nagiosgoodarray)+' Bad:'+str(nagiosbadarray)+' - Disks: OK:'+str(nagiosgooddisk)+' Bad:'+str(nagiosbaddisk))
else:
	if bad:
		# DO NOT MODIFY OUTPUT BELOW
		# Scripts may relies on it
		# https://github.com/eLvErDe/hwraid/issues/99
		print ('\nThere is at least one disk/array in a NOT OPTIMAL state.')
		print ('RAID ERROR - Arrays: OK:'+str(nagiosgoodarray)+' Bad:'+str(nagiosbadarray)+' - Disks: OK:'+str(nagiosgooddisk)+' Bad:'+str(nagiosbaddisk))
		sys.exit(1)

