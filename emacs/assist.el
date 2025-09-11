(require 'gptel)
(require 'dash)

(defun assist/query (query)
  "Opens or finds the right chat buffer for the project, displays that
buffer, adds the query to the buffer, and sends it to gptel"
  (interactive "M")
  (let ((chat-buf (get-buffer-create (assist/buffer-name))))
    (with-current-buffer chat-buf
      (goto-char (point-max))
      (insert "\n\n")
      (insert (assist/full-query query))
      (goto-char (point-max))
      (gptel-send))
    ;; Need to find the right function that opens
    (display-buffer chat-buf)))

(defun assist/full-query (from-user-query)
  (format "%s%s%s"
	  from-user-query
	  (assist/open-buffer-details-string)
	  (assist/project-details-string)))

(defun assist/project-info ()
  "Return some information about the current project to uniquely identify
it from other projects"
  (projectile-acquire-root))

(defun assist/buffer-name ()
  "Create the chat buffer name based on the current buffer's project"
  (format "*%s Chat*"
	  (assist/project-info)))

(defun assist/open-buffers ()
  "Return all open buffers in the focused frame."
  ;; Ensure we are looking only in the current frame:
  (if-let ((frame (selected-frame)))
    ;; Get all buffer names from the given frame:
    (->> frame
     (buffer-list) ; all buffers
     (-map 'get-buffer-window) ; mapped to windows
     (-non-nil) ; remove ones with no window
     (-map 'window-buffer)); back to the buffer itself
    (error "No focused frame found")))

(defun assist/open-buffer-details ()
  "Returns the details (just filename for now) of all the buffers open in
the current frame."
  (->> (assist/open-buffers)
       (-map 'assist/buffer-details)
       (-non-nil)))

(defun assist/open-buffer-details-string ()
  (if-let ((buf-details (assist/open-buffer-details)))
      (format "\n\nHere are all the files within the context of this request:\n[begin open files]\n%s\n[end open files]"
	        (string-join buf-details "\n"))
    ""))

(defun assist/project-details-string ()
  (if-let ((pinfo (assist/project-info)))
      (format "\nThis is the project directory within the context of this request: %s"
	      pinfo)
    ""))

(defun assist/buffer-details (buf)
  "Return the filename associated with the given buffer. nil if the buffer has no file name"
  (when buf
    (let ((file (buffer-file-name buf)))
      (if file
          (expand-file-name file)
        nil))))
