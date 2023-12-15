from pathlib import Path

class Files:
    def __init__(self):
        self.func_list = ["all", "any"]
        self.func_dict = {"all": all, "any": any}
        pass
    
    def check_many(self, multiple, target, func=None):
        if func in self.func_list:
            use_func = self.func_dict[func]
        elif func == None:
            raise Error("pick function 'all' or 'any'")    
        check_ = []
        for i in multiple:
            check_.append(i in target)
        return use_func(check_)
    
    def get_files(self, target_path, suffix, strings=[""], prefix=None, check="all"):
        '''
        target path - (str or pathlib.Path or os.Path) the most shallow searched directory
        suffix - (str) file extension in "*.ext" format
        strings - (list of str) list of strings searched in the file name
        prefix - limit the output list to the file manes starting with prefix
        check - (str) "all" or "any", use the fuction to search for any or all strings in the filename.
        '''
        paths = [str(path) for path in Path(target_path).rglob(suffix) if self.check_many(strings, str(path.name), check)]
        paths.sort()
        if isinstance(prefix, str):
            paths = [path for path in paths if Path(path).name.startswith(prefix)] 
        return paths