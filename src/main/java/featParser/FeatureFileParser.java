
//----------------------------------------------------
// The following code was generated by CUP v0.11a beta 20060608
// Wed Oct 10 16:56:26 EDT 2012
//----------------------------------------------------

package featParser;

import java.io.InputStream;
import java.util.ArrayList;

import java_cup.runtime.DefaultSymbolFactory;
import java_cup.runtime.Symbol;
import java_cup.runtime.SymbolFactory;
import utils.Real;
import data.FeatureFile;
import data.Type;

/** CUP v0.11a beta 20060608 generated parser.
  * @version Wed Oct 10 16:56:26 EDT 2012
  */
public class FeatureFileParser extends java_cup.runtime.lr_parser {

  /** Default constructor. */
  public FeatureFileParser() {super();}

  /** Constructor which sets the default scanner. */
  public FeatureFileParser(java_cup.runtime.Scanner s) {super(s);}

  /** Constructor which sets the default scanner. */
  public FeatureFileParser(java_cup.runtime.Scanner s, java_cup.runtime.SymbolFactory sf) {super(s,sf);}

  /** Production table. */
  protected static final short _production_table[][] = 
    unpackFromStrings(new String[] {
    "\000\036\000\002\002\012\000\002\002\004\000\002\002" +
    "\010\000\002\002\006\000\002\013\003\000\002\013\005" +
    "\000\002\014\003\000\002\014\003\000\002\014\003\000" +
    "\002\014\005\000\002\014\005\000\002\014\005\000\002" +
    "\015\003\000\002\015\003\000\002\015\005\000\002\015" +
    "\005\000\002\003\002\000\002\003\004\000\002\007\007" +
    "\000\002\004\002\000\002\004\004\000\002\010\012\000" +
    "\002\010\014\000\002\005\002\000\002\005\004\000\002" +
    "\011\011\000\002\006\002\000\002\006\011\000\002\016" +
    "\003\000\002\016\003" });

  /** Access to production table. */
  public short[][] production_table() {return _production_table;}

  /** Parse-action table. */
  protected static final short[][] _action_table = 
    unpackFromStrings(new String[] {
    "\000\102\000\004\005\005\001\002\000\004\002\104\001" +
    "\002\000\006\006\ufff1\030\007\001\002\000\006\006\ufff1" +
    "\030\007\001\002\000\004\011\077\001\002\000\004\006" +
    "\011\001\002\000\010\002\uffee\007\uffee\030\013\001\002" +
    "\000\006\002\ufffe\007\045\001\002\000\004\016\016\001" +
    "\002\000\010\002\uffee\007\uffee\030\013\001\002\000\006" +
    "\002\uffed\007\uffed\001\002\000\004\030\020\001\002\000" +
    "\004\017\023\001\002\000\012\012\021\015\ufffd\017\ufffd" +
    "\021\ufffd\001\002\000\004\030\020\001\002\000\010\015" +
    "\ufffc\017\ufffc\021\ufffc\001\002\000\004\011\024\001\002" +
    "\000\004\020\025\001\002\000\010\013\026\014\031\030" +
    "\027\001\002\000\006\012\043\021\ufffa\001\002\000\006" +
    "\012\041\021\ufffb\001\002\000\004\021\034\001\002\000" +
    "\006\012\032\021\ufff9\001\002\000\010\013\026\014\031" +
    "\030\027\001\002\000\004\021\ufff6\001\002\000\012\002" +
    "\uffec\007\uffec\025\035\030\uffec\001\002\000\006\026\036" +
    "\027\037\001\002\000\010\002\uffe5\007\uffe5\030\uffe5\001" +
    "\002\000\010\002\uffe4\007\uffe4\030\uffe4\001\002\000\010" +
    "\002\uffeb\007\uffeb\030\uffeb\001\002\000\010\013\026\014" +
    "\031\030\027\001\002\000\004\021\ufff8\001\002\000\010" +
    "\013\026\014\031\030\027\001\002\000\004\021\ufff7\001" +
    "\002\000\010\002\uffea\010\uffea\030\047\001\002\000\006" +
    "\002\uffff\010\060\001\002\000\004\016\052\001\002\000" +
    "\010\002\uffea\010\uffea\030\047\001\002\000\006\002\uffe9" +
    "\010\uffe9\001\002\000\004\030\020\001\002\000\004\017" +
    "\054\001\002\000\004\004\055\001\002\000\004\030\020" +
    "\001\002\000\004\015\057\001\002\000\010\002\uffe8\010" +
    "\uffe8\030\uffe8\001\002\000\006\002\uffe7\030\062\001\002" +
    "\000\004\002\001\001\002\000\004\020\063\001\002\000" +
    "\006\014\066\030\065\001\002\000\004\021\073\001\002" +
    "\000\006\012\071\021\ufff5\001\002\000\006\012\067\021" +
    "\ufff4\001\002\000\010\013\026\014\031\030\027\001\002" +
    "\000\004\021\ufff2\001\002\000\010\013\026\014\031\030" +
    "\027\001\002\000\004\021\ufff3\001\002\000\004\025\074" +
    "\001\002\000\006\026\036\027\037\001\002\000\006\002" +
    "\uffe7\030\062\001\002\000\004\002\uffe6\001\002\000\004" +
    "\020\100\001\002\000\004\030\020\001\002\000\004\021" +
    "\102\001\002\000\006\006\uffef\030\uffef\001\002\000\004" +
    "\006\ufff0\001\002\000\004\002\000\001\002" });

  /** Access to parse-action table. */
  public short[][] action_table() {return _action_table;}

  /** <code>reduce_goto</code> table. */
  protected static final short[][] _reduce_table = 
    unpackFromStrings(new String[] {
    "\000\102\000\004\002\003\001\001\000\002\001\001\000" +
    "\006\003\007\007\005\001\001\000\006\003\102\007\005" +
    "\001\001\000\002\001\001\000\002\001\001\000\006\004" +
    "\011\010\013\001\001\000\002\001\001\000\002\001\001" +
    "\000\006\004\014\010\013\001\001\000\002\001\001\000" +
    "\004\013\016\001\001\000\002\001\001\000\002\001\001" +
    "\000\004\013\021\001\001\000\002\001\001\000\002\001" +
    "\001\000\002\001\001\000\004\014\027\001\001\000\002" +
    "\001\001\000\002\001\001\000\002\001\001\000\002\001" +
    "\001\000\004\014\032\001\001\000\002\001\001\000\002" +
    "\001\001\000\004\016\037\001\001\000\002\001\001\000" +
    "\002\001\001\000\002\001\001\000\004\014\041\001\001" +
    "\000\002\001\001\000\004\014\043\001\001\000\002\001" +
    "\001\000\006\005\045\011\047\001\001\000\002\001\001" +
    "\000\002\001\001\000\006\005\050\011\047\001\001\000" +
    "\002\001\001\000\004\013\052\001\001\000\002\001\001" +
    "\000\002\001\001\000\004\013\055\001\001\000\002\001" +
    "\001\000\002\001\001\000\004\006\060\001\001\000\002" +
    "\001\001\000\002\001\001\000\004\015\063\001\001\000" +
    "\002\001\001\000\002\001\001\000\002\001\001\000\004" +
    "\014\067\001\001\000\002\001\001\000\004\014\071\001" +
    "\001\000\002\001\001\000\002\001\001\000\004\016\074" +
    "\001\001\000\004\006\075\001\001\000\002\001\001\000" +
    "\002\001\001\000\004\013\100\001\001\000\002\001\001" +
    "\000\002\001\001\000\002\001\001\000\002\001\001" });

  /** Access to <code>reduce_goto</code> table. */
  public short[][] reduce_table() {return _reduce_table;}

  /** Instance of action encapsulation class. */
  protected CUP$FeatureFileParser$actions action_obj;

  /** Action encapsulation object initializer. */
  protected void init_actions()
    {
      action_obj = new CUP$FeatureFileParser$actions(this);
    }

  /** Invoke a user supplied parse action. */
  public java_cup.runtime.Symbol do_action(
    int                        act_num,
    java_cup.runtime.lr_parser parser,
    java.util.Stack            stack,
    int                        top)
    throws java.lang.Exception
  {
    /* call code in generated class */
    return action_obj.CUP$FeatureFileParser$do_action(act_num, parser, stack, top);
  }

  /** Indicates start state. */
  public int start_state() {return 0;}
  /** Indicates start production. */
  public int start_production() {return 1;}

  /** <code>EOF</code> Symbol index. */
  public int EOF_sym() {return 0;}

  /** <code>error</code> Symbol index. */
  public int error_sym() {return 1;}



	public static void main(String args[]) throws Exception {
		SymbolFactory sf = new DefaultSymbolFactory();
		ff = new FeatureFile();
		if (args.length==0) new FeatureFileParser(new FeatureFileScanner(System.in,sf),sf).parse();
		else new FeatureFileParser(new FeatureFileScanner(new java.io.FileInputStream(args[0]),sf),sf).parse();
	}
	
	public static FeatureFileParser createParser(String filename) throws Exception {
		SymbolFactory sf = new DefaultSymbolFactory();
		ff = new FeatureFile();
		return new FeatureFileParser(new FeatureFileScanner(new java.io.FileInputStream(filename),sf),sf);
	}
	
	public static FeatureFileParser createParser(InputStream is) throws Exception {
        SymbolFactory sf = new DefaultSymbolFactory();
        ff = new FeatureFile();
        return new FeatureFileParser(new FeatureFileScanner(is,sf),sf);
    }
	
	public FeatureFile parseFile() throws Exception {
		//System.out.println("Debug parsing");
		parse();
		return ff;
	}
	public static FeatureFile ff;
	
	public void syntax_error(java_cup.runtime.Symbol current) {
		report_error("Syntax error (" + current.sym + ")", current);
	}
	public void report_fatal_error(String message, Object info) {
		System.out.println(": "+message);
    
		if ( !(info instanceof Symbol) ) return;
		Symbol symbol = (Symbol) info;
		System.out.println(" left "+symbol.left+", right "+symbol.right);
    
		if ( symbol.left < 0 || symbol.right < 0 ) return;
    
		System.out.println(" at line "+symbol.left+", column "+symbol.right);
	}
	public void report_error(String message, Object info) {
		System.out.println(": "+message);
    
		if ( !(info instanceof Symbol) ) return;
		Symbol symbol = (Symbol) info;
		System.out.println(" left "+symbol.left+", right "+symbol.right);
    
		if ( symbol.left < 0 || symbol.right < 0 ) return;
    
		System.out.println(" at line "+symbol.left+", column "+symbol.right);
	}

}

/** Cup generated class to encapsulate user supplied action code.*/
class CUP$FeatureFileParser$actions {
  private final FeatureFileParser parser;

  /** Constructor */
  CUP$FeatureFileParser$actions(FeatureFileParser parser) {
    this.parser = parser;
  }

  /** Method with the actual generated action code. */
  public final java_cup.runtime.Symbol CUP$FeatureFileParser$do_action(
    int                        CUP$FeatureFileParser$act_num,
    java_cup.runtime.lr_parser CUP$FeatureFileParser$parser,
    java.util.Stack            CUP$FeatureFileParser$stack,
    int                        CUP$FeatureFileParser$top)
    throws java.lang.Exception
    {
      /* Symbol object for return from actions */
      java_cup.runtime.Symbol CUP$FeatureFileParser$result;

      /* select the action based on the action number */
      switch (CUP$FeatureFileParser$act_num)
        {
          /*. . . . . . . . . . . . . . . . . . . .*/
          case 29: // double_num ::= NUMBER 
            {
              Double RESULT =null;
		int numleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int numright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		Integer num = (Integer)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 RESULT = new Double(num); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("double_num",12, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 28: // double_num ::= DOUB 
            {
              Double RESULT =null;
		int dblleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int dblright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		Double dbl = (Double)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 RESULT = dbl; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("double_num",12, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 27: // weights ::= IDENT LSB ident_dollar_set RSB EQL double_num weights 
            {
              Object RESULT =null;
		int inleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)).left;
		int inright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)).right;
		String in = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)).value;
		int insleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).left;
		int insright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).right;
		ArrayList<String> ins = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).value;
		int dblleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).left;
		int dblright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).right;
		Double dbl = (Double)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).value;
		 
												 String name = in+"*";
											     String valName = "";
											     for(String val : ins) {
													 if(val=="$") val = "X";
												     valName=valName+"_"+val;
												 }
												name+=valName;
												parser.ff.setWeight(name,new Real(dbl)); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("weights",4, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 26: // weights ::= 
            {
              Object RESULT =null;
		 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("weights",4, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 25: // rel_declaration ::= IDENT LB ident_set RB ARROW ident_set SEMI 
            {
              Object RESULT =null;
		int inleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)).left;
		int inright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)).right;
		String in = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)).value;
		int ins1left = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).left;
		int ins1right = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).right;
		ArrayList<String> ins1 = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).value;
		int ins2left = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).left;
		int ins2right = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).right;
		ArrayList<String> ins2 = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).value;
		 parser.ff.addRelation(in, ins1,ins2); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("rel_declaration",7, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-6)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 24: // rel_d ::= rel_declaration rel_d 
            {
              Object RESULT =null;
		 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("rel_d",3, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 23: // rel_d ::= 
            {
              Object RESULT =null;
		 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("rel_d",3, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 22: // feature_declaration ::= IDENT LB ident_set RB ASSIGN LSB ident_star_set RSB EQL double_num 
            {
              Object RESULT =null;
		int inleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-9)).left;
		int inright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-9)).right;
		String in = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-9)).value;
		int insleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)).left;
		int insright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)).right;
		ArrayList<String> ins = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)).value;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-3)).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-3)).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-3)).value;
		int dblleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int dblright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		Double dbl = (Double)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 parser.ff.addFeatureTypenames(in,ins,iss); 
												String name = in+"*";
											    String valName = "";
											    for(String val : iss) {
													if(val=="$") val = "X";
												    valName=valName+"_"+val;
												}
												name+=valName;
												parser.ff.setWeight(name,new Real(dbl));
												
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("feature_declaration",6, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-9)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 21: // feature_declaration ::= IDENT LB ident_set RB ASSIGN LSB ident_star_set RSB 
            {
              Object RESULT =null;
		int inleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)).left;
		int inright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)).right;
		String in = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)).value;
		int insleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-5)).left;
		int insright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-5)).right;
		ArrayList<String> ins = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-5)).value;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).value;
		 parser.ff.addFeatureTypenames(in,ins,iss); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("feature_declaration",6, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 20: // features_d ::= feature_declaration features_d 
            {
              Object RESULT =null;
		  
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("features_d",2, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 19: // features_d ::= 
            {
              Object RESULT =null;
		  
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("features_d",2, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 18: // type_declaration ::= IDENT ASSIGN LSB ident_set RSB 
            {
              Object RESULT =null;
		int inleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).left;
		int inright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).right;
		String in = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)).value;
		int insleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).left;
		int insright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).right;
		ArrayList<String> ins = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).value;
		   
											Type t;
											t = new Type(in);
											for(String ident: ins) 
												t.addValue(ident);
											parser.ff.addType(in,t); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("type_declaration",5, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-4)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 17: // types_d ::= type_declaration types_d 
            {
              Object RESULT =null;
		  
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("types_d",1, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 16: // types_d ::= 
            {
              Object RESULT =null;
		 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("types_d",1, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 15: // ident_dollar_set ::= DOLLAR COMMA ident_star_set 
            {
              ArrayList<String> RESULT =null;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 iss.add(0,"$"); RESULT=iss; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_dollar_set",11, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 14: // ident_dollar_set ::= IDENT COMMA ident_star_set 
            {
              ArrayList<String> RESULT =null;
		int indleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).left;
		int indright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).right;
		String ind = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).value;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 iss.add(0,ind); RESULT=iss; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_dollar_set",11, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 13: // ident_dollar_set ::= DOLLAR 
            {
              ArrayList<String> RESULT =null;
		 RESULT = new ArrayList<String>(); RESULT.add("$"); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_dollar_set",11, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 12: // ident_dollar_set ::= IDENT 
            {
              ArrayList<String> RESULT =null;
		int indleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int indright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		String ind = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 RESULT = new ArrayList<String>(); RESULT.add(ind); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_dollar_set",11, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 11: // ident_star_set ::= DOLLAR COMMA ident_star_set 
            {
              ArrayList<String> RESULT =null;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 iss.add(0,"$"); RESULT=iss; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_star_set",10, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 10: // ident_star_set ::= STAR COMMA ident_star_set 
            {
              ArrayList<String> RESULT =null;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 iss.add(0,"*"); RESULT=iss; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_star_set",10, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 9: // ident_star_set ::= IDENT COMMA ident_star_set 
            {
              ArrayList<String> RESULT =null;
		int indleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).left;
		int indright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).right;
		String ind = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).value;
		int issleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int issright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		ArrayList<String> iss = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 iss.add(0,ind); RESULT=iss; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_star_set",10, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 8: // ident_star_set ::= DOLLAR 
            {
              ArrayList<String> RESULT =null;
		 RESULT = new ArrayList<String>(); RESULT.add("$"); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_star_set",10, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 7: // ident_star_set ::= STAR 
            {
              ArrayList<String> RESULT =null;
		 RESULT = new ArrayList<String>(); RESULT.add("*"); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_star_set",10, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 6: // ident_star_set ::= IDENT 
            {
              ArrayList<String> RESULT =null;
		int indleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int indright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		String ind = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 RESULT = new ArrayList<String>(); RESULT.add(ind); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_star_set",10, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 5: // ident_set ::= IDENT COMMA ident_set 
            {
              ArrayList<String> RESULT =null;
		int indleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).left;
		int indright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).right;
		String ind = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)).value;
		int isleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int isright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		ArrayList<String> is = (ArrayList<String>)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 is.add(0,ind); RESULT=is; 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_set",9, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-2)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 4: // ident_set ::= IDENT 
            {
              ArrayList<String> RESULT =null;
		int indleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).left;
		int indright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()).right;
		String ind = (String)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.peek()).value;
		 RESULT = new ArrayList<String>(); RESULT.add(ind); 
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("ident_set",9, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 3: // program ::= TYPES types_d FEATURES features_d 
            {
              Object RESULT =null;
		  
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("program",0, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-3)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 2: // program ::= TYPES types_d FEATURES features_d RELATIONS rel_d 
            {
              Object RESULT =null;
		  
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("program",0, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-5)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 1: // $START ::= program EOF 
            {
              Object RESULT =null;
		int start_valleft = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).left;
		int start_valright = ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).right;
		Object start_val = (Object)((java_cup.runtime.Symbol) CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)).value;
		RESULT = start_val;
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("$START",0, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-1)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          /* ACCEPT */
          CUP$FeatureFileParser$parser.done_parsing();
          return CUP$FeatureFileParser$result;

          /*. . . . . . . . . . . . . . . . . . . .*/
          case 0: // program ::= TYPES types_d FEATURES features_d RELATIONS rel_d WEIGHTS weights 
            {
              Object RESULT =null;
		 
													
												
              CUP$FeatureFileParser$result = parser.getSymbolFactory().newSymbol("program",0, ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.elementAt(CUP$FeatureFileParser$top-7)), ((java_cup.runtime.Symbol)CUP$FeatureFileParser$stack.peek()), RESULT);
            }
          return CUP$FeatureFileParser$result;

          /* . . . . . .*/
          default:
            throw new Exception(
               "Invalid action number found in internal parse table");

        }
    }
}

