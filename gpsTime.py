import numpy as np
from math import modf

import datetime as dt
import calendar

def cal2jd(yr,mn,dy) :
    """
    CAL2JD  Converts calendar date to Julian date using algorithm
    from "Practical Ephemeris Calculations" by Oliver Montenbruck
    (Springer-Verlag, 1989). Uses astronomical year for B.C. dates
    (2 BC = -1 yr). 

    Input:
        yr  : YYYY (int)
        mn  : MM 01 to 12 (int)
        day : DD 01 to 31 (int)

    Output:
        jd : julian date (float)

    """

    if mn > 2:
        y = yr
        m = mn
    else:
        y = yr - 1
        m = mn + 12

    date1=4.5+31.*(10.+12.*1582.)   # Last day of Julian calendar (1582.10.04 Noon)
    date2=15.5+31.*(10.+12.*1582.)  # First day of Gregorian calendar (1582.10.15 Noon)
    date=dy+31.*(mn+12.*yr)

    if date <= date1:
        b = -2
    elif date >= date2 :
        b = np.fix(y/400.) - np.fix(y/100.)
    else:
        #warning('Dates between October 5 & 15, 1582 do not exist');
        return

    if y > 0:
        jd = np.fix(365.25*y) + np.fix(30.6001*(m+1)) + b + 1720996.5 + dy
    else:
        jd = np.fix(365.25*y-0.75) + np.fix(30.6001*(m+1)) + b + 1720996.5 + dy

    return jd

def yyyydoy2jd(year,doy,hh=0,mm=0,ss=0.0):
    """
    yyyydoy2jd Take a year, day-of-year, etc and convert it into a julian day

    Usage:  jd = yyyydoy2jd(year,doy,hh,mm,ss)

    Input:  year - 4 digit integer
            doy  - 3 digit, or less integer, (1 <= doy <= 366)
            hh   - 2 digit, or less int, (0 <= hh < 24) (not required)
            mm   - 2 digit, or less int,(0 <= ss < 60) (not required)
            ss   - float (not required) 

    Output: 'jd' (float) 

    """
    #
    # need to split seconds into two components
    # sec => 2 digit, or less int, (0 <= ss < 60)
    # ms =>  int 0 <= ms < 1,000,000
    #
    ms,sec  = modf(float(ss)) 
    ms = ms * 10e5

    dto = dt.datetime(int(year),01,01,int(hh),int(mm),int(sec),int(ms))
    dto = dto + dt.timedelta(days=(int(doy) - 1))

    mn = dto.month
    dy = dto.day

    jd = cal2jd(int(year),int(mn),int(dy)) 
    jd = jd + float(hh)/24. + float(mm)/60./24. + float(sec)/3600./24.
    return jd - 2400000.5 

def jd2gps(jd):
    """
      JD2GPS  Converts Julian date to GPS week number (since
        1980.01.06) and seconds of week. 
      Usage:   [gpsweek,sow,rollover]=jd2gps(jd)
      Input:   jd       - Julian date
      Output:  gpsweek  - GPS week number
               sow      - seconds of week since 0 hr, Sun.
               rollover - number of GPS week rollovers (modulus 1024)

      Copyright (c) 2011, Michael R. Craymer
      All rights reserved.
      Email: mike@craymer.com
    """

    jdgps = cal2jd(1980,1,6);    # beginning of GPS week numbering
    nweek = int(np.fix((jd-jdgps)/7.))
    sow = (jd - (jdgps+nweek*7)) * 3600*24
    rollover = np.fix(nweek/1024)  # rollover every 1024 weeks
    gpsweek = int(nweek)

# rollover is being returned as an array?
# should just be an int

    return gpsweek,sow,rollover 


def jd2cal(jd):
    """
      JD2CAL  Converts Julian date to calendar date using algorithm
        from "Practical Ephemeris Calculations" by Oliver Montenbruck
        (Springer-Verlag, 1989). Must use astronomical year for B.C.
        dates (2 BC = -1 yr). Non-vectorized version. See also CAL2JD,
        DOY2JD, GPS2JD, JD2DOW, JD2DOY, JD2GPS, JD2YR, YR2JD.
      Usage:   [yr, mn, dy]=jd2cal(jd)
      Input:   jd - Julian date
      Output:  yr - year of calendar date
               mn - month of calendar date
               dy - day of calendar date (including decimal)

      Copyright (c) 2011, Michael R. Craymer
      All rights reserved.
      Email: mike@craymer.com
    """
    a = np.fix(jd+0.5)

    if a < 2299161. :
        c = a + 1524.
    else:
        b = np.fix( (a-1867216.25) / 36524.25 )
        c = a + b - np.fix(b/4.) + 1525.

    d = np.fix( (c-122.1)/365.25 )
    e = np.fix(365.25*d)
    f = np.fix( (c-e) / 30.6001 )
    dy = c - e - np.fix(30.6001*f) + np.remainder((jd+0.5),a)
    mn = f - 1. - 12. * np.fix(f/14.)
    yr = d - 4715. - np.fix( (7.+mn)/10. )

    return  yr,mn,dy 

def jd2doy(jd):
    """
      JD2DOY  Converts Julian date to year and day of year.
      Usage:   [doy,yr]=jd2doy(jd)
      Input:   jd  - Julian date
      Output:  doy - day of year
               yr  - year

      Copyright (c) 2011, Michael R. Craymer
      All rights reserved.
      Email: mike@craymer.com
    """
    [yr,mn,dy] = jd2cal(jd)
    doy = jd - cal2jd(yr,1,0)

    # MM ensure the doy is 0 padded
    doy = "%03d" % doy
    return yr, doy

def yyyy2yy(year):
    """
      yy = yyyy2yy(YYYY)

      return the yy form of YYYY
        yy - last two digits of YYYY
           - returned as an int

      very messy hack
    """
    yy = int( str(int(year))[-2] + str(int(year))[-1] )
    return(yy)

def dateTime2gpssow(dt):
    """
    dateTime2gpssow Converts a datetime object into gps week, 
                    and gps seconds of week

    Usage: week,sow = dateTime2gpssow(dateTime)

    Input: dt - python datetime object

    Output: week - gps week (int)
            sow  - seconds into gpsweek since 0 hr, Sunday (float)

    """
    day = dt.day + dt.hour/24. + dt.minute/1440. + dt.second/86400.
    jd = cal2jd(dt.year,dt.month,day)
    week, sow, rollover = jd2gps(jd)
    return week, sow

def ydhms2dt(year,doy,hh,mm,ss):
    """
    ydhms2dt Take a year, day-of-year, etc and convert it into a date time object

    Usage:  dto = ydhms2dt(year,day,hh,mm,ss)

    Input:  year - 4 digit integer
            doy  - 3 digit, or less integer, (1 <= doy <= 366)
            hh   - 2 digit, or less int, (0 <= hh < 24)
            mm   - 2 digit, or less int,(0 <= ss < 60)
            ss   - float 

    Output: 'dto' a date time object

    """
    #
    # need to split seconds into two components
    # sec => 2 digit, or less int, (0 <= ss < 60)
    # ms =>  int 0 <= ms < 1,000,000
    ms,sec  = modf(float(ss)) 
    ms = ms * 10e5

    dto = dt.datetime(int(year),01,01,int(hh),int(mm),int(sec),int(ms))
    dto = dto + dt.timedelta(days=(int(doy) - 1))
    return dto 

def ymdhms2dt(year,month,day,hh,mm,ss):
    """
    ymhms2dt Take a year, day-of-year, etc and convert it into a date time object

    Usage:  dto = ymdhms2dt(year,month,day,hh,mm,ss)

    Input:  year - 4 digit integer
            month - integer, (1 => January)
            day  - integer 
            hh   - 2 digit, or less int, (0 <= hh < 24)
            mm   - 2 digit, or less int,(0 <= ss < 60)
            ss   - float 

    Output: 'dto' a date time object

    """
    #
    # need to split seconds into two components
    # sec => 2 digit, or less int, (0 <= ss < 60)
    # ms =>  int 0 <= ms < 1,000,000
    ms,sec  = modf(float(ss)) 
    ms = ms * 10e5

    dto = dt.datetime(int(year),int(month),int(day),int(hh),int(mm),int(sec),0)
    #dto = dt.datetime(int(year),int(month),int(day),int(hh),int(mm),int(sec),int(ms))
#dt.date(int(year),int(month),int(day)) 
    #01,01,int(hh),int(mm),int(sec),int(ms))
    #dto = dto + dt.timedelta(hours= int(hh),minutes=int(mm),seconds=int(sec))
    return dto 

def jd2mdt(jd):
    """
    jd2mdt Take a julian date and convert it into a matplotlib date time stamp
               All matplotlib date plotting is done by converting date instances into
               days since the 0001-01-01 UTC

    Usage:  mp_ts = jd2mdt(jd)

    Input: jd julian date

    Output: 'mp_ts' (float) 
            a matplot lib time stamp which is days from 0001-01-01

    """ 

    #ms,sec  = modf(float(ss)) 
    #ms = ms * 10e5

    year,mon,d = jd2cal(jd)

    day = int(np.fix(d))

    h = (d - float(day)) * 24.
    hh = int(np.fix(h))

    m = (h - float(hh)) * 60.
    mm = int(np.fix(m))

    s = (m - float(mm)) * 60.
    sec = int(np.fix(s))
 
    ms = 0
    dto = dt.datetime(int(year),int(mon),int(day),int(hh),int(mm),int(sec),int(ms))

    mp_epoch = dt.datetime(1, 1, 1)
    DAY = 86400
    td = dto - mp_epoch
    mp_ts = td.days + 1 + (1000000 * td.seconds + td.microseconds) / 1e6 / DAY
    return mp_ts

def ydhms2mdt(year,doy,hh,mm,ss):
    """
    ydhms2mdt Take a year, day-of-year, etc and convert it into a matplotlib date
               All matplotlib date plotting is done by converting date instances into
               days since the 0001-01-01 UTC

    Usage:  mp_ts = ydhms2dt(year,day,hh,mm,ss)

    Input:  year - 4 digit integer
            doy  - 3 digit, or less integer, (1 <= doy <= 366)
            hh   - 2 digit, or less int, (0 <= hh < 24)
            mm   - 2 digit, or less int,(0 <= ss < 60)
            ss   - float 

    Output: 'mp_ts' (float) 
            a matplot lib time stamp which is days from 0001-01-01

    """
    #
    # need to split seconds into two components
    # sec => 2 digit, or less int, (0 <= ss < 60)
    # ms =>  int 0 <= ms < 1,000,000
    ms,sec  = modf(float(ss)) 
    ms = ms * 10e5

    dto = dt.datetime(int(year),01,01,int(hh),int(mm),int(sec),int(ms))
    dto = dto + dt.timedelta(days=(int(doy) - 1))

    mp_epoch = dt.datetime(1, 1, 1)
    DAY = 86400
    td = dto - mp_epoch
    mp_ts = td.days + 1 + (1000000 * td.seconds + td.microseconds) / 1e6 / DAY
    return mp_ts

def ydhms2decyr(year,doy,hh=0,mm=0,ss=0.0):
    """
    ydhms2decyr(year,doy,hh,mm,ss)

    Convert from Year, Day-of-year to decimal year

    """
    #ms,sec  = modf(float(ss))
    #ms = ms * 10e5
    #dto = dt.datetime(int(year),,,int(hh),int(mm),int(sec),int(ms))
    #dto = dto + dt.timedelta(days=(int(doy) - 1))

    if calendar.isleap(int(year)):
        dec_yr = float(year) + (float(doy) - 1.)/366. + float(hh)/24./366 + float(mm)/60./24./366. + float(ss)/86400./366.
    else:
        dec_yr = float(year) + (float(doy) - 1.)/365. + float(hh)/24./365 + float(mm)/60./24./365. + float(ss)/86400./365.

    
    return dec_yr

# TODO check this starts before 1970, and end before 2050
#      check it returns the time in ms
def dt2unix(dto):
    """
    dt2unix : Convert a datetime object to a UNIX time stamp

    Usage: unixTS = dt2unix(dto)

    Input:dto A datetime object

    Output: a unix time stamp (int)

    """
    return calendar.timegm(dto.utctimetuple())

def unix2dt(unix_timestamp):
    dto = dt.datetime.utcfromtimestamp(int(unix_timestamp))
    return dto

def dt2validFrom(dto):
    """
    dt2validFrom(dto)
    Return the values needed to form a valid from string, from a date time object

    """
    yyyy = dto.strftime("%Y")
    MM = dto.strftime("%m")
    dd = dto.strftime("%d")
    hh = dto.strftime("%H")
    mm = dto.strftime("%M")
    ss = dto.strftime("%S")
    ms = dto.strftime("%f")
    return yyyy, MM, dd, hh, mm, ss, ms

# modified julian day
# ymdhms -> modJ
# gps week -> modj
# modj -> gps week
# 
#=========================

if __name__ == "__main__":

    startymdhms = cal2jd(2012,01,01+(00/24)+(00/(24*60))+(00.0000/(24*3600)))
    (year,doy) = jd2doy(startymdhms)
    print(year,doy)
    # now obatain the yy version of YYYY
    yy = yyyy2yy(year)
    print(yy)
    yy = yyyy2yy(2012.7)
    print(yy)

    print("Test of ydhms2dt( 2013, 200, 13, 23, 32.567)")
    dt = ydhms2dt(2013,200,13,23,32.567)
    print(dt)
#print(jd)
#2455927.5

#(gpsweek,sow,rollover) = gt.jd2gps(jd)
#print(gpsweek,sow,rollover)
#(1669, 0.0, array(1.0))
#=> should be (1669, 0.0, 1.0)

