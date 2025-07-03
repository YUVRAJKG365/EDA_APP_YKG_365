# tracker.py
class AppTracker:
    def __init__(self):
        self.logs = []
        self.current_section = None
        self.tab_stack = []  # Stack to support nested tabs and sub-tabs

    def log_section(self, section):
        """
        Log a new section visit and reset tab stack.
        """
        self.current_section = section
        self.tab_stack = []
        self.logs.append(f"Visited section: {section}")

    def log_tab(self, tab):
        """
        Log opening a top-level tab; resets sub-tabs.
        """
        self.tab_stack = [tab]
        self.logs.append(f"Opened tab: {tab} in section: {self.current_section}")

    def log_subtab(self, subtab):
        """
        Log opening a nested sub-tab under the current tab.
        """
        self.tab_stack.append(subtab)
        full_tab_path = " > ".join(self.tab_stack)
        self.logs.append(
            f"Opened sub-tab: {full_tab_path} in section: {self.current_section}"
        )

    def log_operation(self, message):
        """
        Log an operation, including section and full tab/sub-tab path.
        """
        location = self.current_section if self.current_section else ""
        if self.tab_stack:
            location += " > " + " > ".join(self.tab_stack)
        self.logs.append(f"In {location}: {message}")

    def get_context(self):
        """
        Return the full log context as a single string.
        """
        return "\n".join(self.logs)

    def clear(self):
        """
        Clear all logs and reset state.
        """
        self.logs.clear()
        self.current_section = None
        self.tab_stack = []
