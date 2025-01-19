import code
import operator
import sys

class StateKernel(code.InteractiveConsole):
    """
    StateKernel is a subclass of InteractiveConsole that provides a custom REPL.
    
    This subclass builds on InteractiveInterpreter and adds prompting using the familiar
    sys.ps1 and sys.ps2, input buffering, and improved operator handling within the REPL environment.
    """

    # Define some basic arithmetic operators
    OPERATORS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }

    def __init__(self, locals=None, filename="<console>", local_exit=False):
        """
        Initialize a StateKernel instance.

        Parameters:
        - locals: A mapping to use as the namespace for code execution; defaults to a new dictionary.
        - filename: The filename from which the source was read; defaults to "<console>".
        - local_exit: If True, exit() and quit() will not raise SystemExit; they just return to the calling code.
        """
        super().__init__(locals=locals, filename=filename)
        self.local_exit = local_exit

    def runcode(self, code_obj):
        """
        Execute a code object and handle exceptions, excluding SystemExit if local_exit is False.
        
        This method overrides the parent class method to provide more customized exception handling.
        
        Parameters:
        - code_obj: The compiled code object to be executed.
        """
        try:
            exec(code_obj, self.locals)
        except Exception as e:
            if isinstance(e, SystemExit) and self.local_exit:
                print("Exit attempted but local_exit is True. Returning to REPL.")
            else:
                self.showtraceback()

    def showtraceback(self):
        """
        Display the exception that just occurred, without showing the first stack item since
        it is within the interpreter object implementation.
        
        This method overrides the parent class method to customize the output.
        """
        type, value, tb = sys.exc_info()
        print(f"Exception of type {type.__name__} occurred with message: {value}")
        super().showtraceback()

    def showsyntaxerror(self, filename=None):
        """
        Display the syntax error that just occurred. 
        
        Overridden from the parent class to provide a custom message format.
        
        Parameters:
        - filename: Optionally, specify a custom filename for the syntax error message.
        """
        super().showsyntaxerror(filename=filename)
    
    def interact(self, banner=None, exitmsg=None):
        """
        Emulate the interactive Python console.

        Parameters:
        - banner: The banner printed before the first interaction; defaults to None.
        - exitmsg: The message printed on exit; defaults to None (prints a default message).
        """
        if banner is None:
            banner = "Welcome to the StateKernel custom shell. Type exit() or quit() to exit."
        if exitmsg is None:
            exitmsg = "Goodbye from StateKernel!"
        super().interact(banner=banner, exitmsg=exitmsg)

    def push(self, line):
        """
        Push a line of source text to the interpreter.

        The line should not have a trailing newline; it may contain internal newlines.

        Parameters:
        - line: The source text to be executed.

        Returns:
        - A boolean indicating if more input is required (True) or if the line was processed (False).
        """
        try:
            tokens = line.split()
            if len(tokens) == 3 and tokens[1] in self.OPERATORS:
                operand1 = float(tokens[0])
                operand2 = float(tokens[2])
                operation = self.OPERATORS[tokens[1]]
                print(f"Result: {operation(operand1, operand2)}")
                return False
        except (ValueError, KeyError) as e:
            print(f"Invalid input or operation: {e}. Executing line normally.")
        
        return super().push(line)

import readline
from glob import glob

def completer():
    class tabCompleter(object):
        def pathCompleter(self,text,state):
            line   = readline.get_line_buffer().split()
            return [x for x in glob(text+'*')][state]
        def createListCompleter(self,ll):
            pass
            def listCompleter(text,state):
                line   = readline.get_line_buffer()
                if not line:
                    return None
                else:
                    return [c + " " for c in ll if c.startswith(line)][state]
            self.listCompleter = listCompleter
    t = tabCompleter()
                            # tool command
    t.createListCompleter(["clear", "exit", "banner","exec","restart", "upgrade", 'search'
                            # modules
                            # auxiliary modules
                             ,"use auxiliary/gather/ip_gather","use auxiliary/gather/ip_lookup", "use auxiliary/core/pyconverter"
                            # exploit modules
                             ,"use exploit/windows/ftp/ftpshell_overflow", "use exploit/android/login/login_bypass", "use exploit/windows/http/oracle9i_xdb_pass"])
    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(t.listCompleter)


class interpreter(object):
    def search_module(query):
        import utilities.files
        from core.module_obtainer import obtainer
        reload(utilities.files)
        from utilities.files import count_subfolders_files
        print('{:44}{:29}'.format('\nModules','Description'))
        print('{:43}{:29}'.format('-------','-----------'))
        count_subfolders_files('modules')
        unfiltered_result = []
        result = []
        for first_result in count_subfolders_files.result:
            for second_result in first_result:
                unfiltered_result.append(''.join(second_result))
                for final_result in unfiltered_result:
                    if final_result.endswith('.pyc') or final_result.startswith('__init__') or not final_result.endswith('.py'):
                        unfiltered_result.remove(final_result)
        for filt_result in unfiltered_result:
            result.append(filt_result.split('.')[0])
        try:
            for i in range(0,10):
                for root, dirs, files in os.walk('modules'):
                    if query in root+result[i]:
                        if result[i]+'.py' in files:
                            path = os.path.join(root, result[i]+'.py')
                            path = path.split('/')
                            path_first_index = path[0]
                            path.remove(path_first_index)
                            path = '/'.join(path)
                            if obtainer.description_obtainer(obtainer,path.split('.py')[0]):
                                print('{:44}{:44}'.format(path.split('.py')[0],obtainer.info['description']))
        except IndexError:
            pass
        except ImportError:
            pass

        print('')

    def start_interpreter(self):
        completer()
        from core.validator import validator
        while True:
            try:
                self.main_ask = input(underline("PySploit") + " >> ").split()
                validator(self.main_ask).validate_interpreter_mode()
            except (KeyboardInterrupt,EOFError):
                print('\n' + red('\n[!]') + green(' type') + gray(' exit') + ' to close the program\n')

import readline
from glob import glob

def completer():
    class tabCompleter(object):
        def pathCompleter(self,text,state):
            line   = readline.get_line_buffer().split()
            return [x for x in glob(text+'*')][state]
        def createListCompleter(self,ll):
            pass
            def listCompleter(text,state):
                line   = readline.get_line_buffer()
                if not line:
                    return None
                else:
                    return [c + " " for c in ll if c.startswith(line)][state]
            self.listCompleter = listCompleter
    t = tabCompleter()
                            # tool command
    t.createListCompleter(["clear", "exit", "banner","exec","restart", "upgrade", 'search'
                            # modules
                            # auxiliary modules
                             ,"use auxiliary/gather/ip_gather","use auxiliary/gather/ip_lookup", "use auxiliary/core/pyconverter"
                            # exploit modules
                             ,"use exploit/windows/ftp/ftpshell_overflow", "use exploit/android/login/login_bypass", "use exploit/windows/http/oracle9i_xdb_pass"])
    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(t.listCompleter)



if __name__ == "__main__":
    # Initialize and start the custom interactive shell
    kernel = StateKernel()
    kernel.interact()