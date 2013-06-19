package utils;

import java.util.HashMap;
import java.util.Map.Entry;

public class Profiler {
	private static HashMap<String, Long> runningProcesses = new HashMap<String, Long>();
	private static HashMap<String, Long> totalProcessTimes = new HashMap<String, Long>();
	private static long startTime = -1;

	public static void start(){
		startTime = System.currentTimeMillis();
	}
	public static void startProcess(String processName){
		runningProcesses.put(processName, System.currentTimeMillis());
		if(!totalProcessTimes.containsKey(processName))
			totalProcessTimes.put(processName, new Long(0));
	}
	public static long endProcess(String processName){
		Long time = runningProcesses.remove(processName);
		if(time==null)
			return -1;
		long elapsedTime = System.currentTimeMillis()-time;
		totalProcessTimes.put(processName,totalProcessTimes.get(processName)+elapsedTime);
		return elapsedTime;
	}
	public static void endProcessPrint(String processName){
		Long elapsedTime = endProcess(processName);
		if(elapsedTime<0){
			System.out.println("PROFILER: "+processName+" no start time.");
		}else{
			System.out.println("PROFILER: "+processName+" ran in "+formatTime(elapsedTime));
		}
	}
	
	public static String formatTime(long timeMillis){
		long time = timeMillis / 1000;  
		String seconds = Integer.toString((int)(time % 60));  
		String minutes = Integer.toString((int)((time % 3600) / 60));  
		String hours = Integer.toString((int)(time / 3600));  
		for (int i = 0; i < 2; i++) {  
			if (seconds.length() < 2) {  
				seconds = "0" + seconds;  
			}  
			if (minutes.length() < 2) {  
				minutes = "0" + minutes;  
			}  
			if (hours.length() < 2) {  
				hours = "0" + hours;  
			}  
		}
		return hours+":"+minutes+":"+seconds;
	}
	
	public static void printProcessTimes(){
		String total = (startTime>0?" total ("+formatTime(System.currentTimeMillis()-startTime)+")":"");
		System.out.println("PROFILER ----- Process run times "+total+"------");
		for(Entry<String,Long> pEntry:totalProcessTimes.entrySet()){
			System.out.println("PROFILER:" +pEntry.getKey()+" in "+formatTime(pEntry.getValue()));			
		}
		System.out.println("PROFILER -----------------------------");
	}
}
