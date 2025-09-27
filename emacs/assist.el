;;; assist.el --- Emacs interface for Assist server -*- lexical-binding: t -*-

;; Copyright (C) 2024

;; Author: Assist Contributors
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: convenience, tools
;; URL: https://github.com/user/assist

;;; Commentary:

;; Assist.el provides an Emacs interface to the Assist server.
;; It allows users to have conversations with an AI assistant
;; directly from Emacs buffers using org-mode.

;;; Code:

(require 'org)
(require 'json)
(require 'url)
(require 'cl-lib)

;;; Customization

(defgroup assist nil
  "Assist server integration for Emacs."
  :group 'convenience
  :prefix "assist/")

(defcustom assist-server-url "http://localhost:5000"
  "URL of the Assist server."
  :type 'string
  :group 'assist)

(defcustom assist-buffer-prefix "Assist: "
  "Prefix for Assist chat buffers."
  :type 'string
  :group 'assist)

;;; Variables

(defvar assist--active-requests (make-hash-table :test 'equal)
  "Hash table tracking active requests by buffer name.")

(defvar assist--status "assist"
  "Current status string for mode line.")

(defvar assist--last-chat-buffer nil
  "Name of the last used chat buffer.")

;;; Mode line integration

(defvar assist-mode-line-format
  '(:eval (when assist-minor-mode
            (format " %s" assist--status))))

;;; Utility functions

(defun assist--buffer-is-assist-p (&optional buffer)
  "Check if BUFFER is an Assist chat buffer."
  (let ((buf (or buffer (current-buffer))))
    (string-prefix-p assist-buffer-prefix (buffer-name buf))))

(defun assist--get-chat-buffers ()
  "Return list of existing Assist chat buffers."
  (seq-filter (lambda (buf)
                (string-prefix-p assist-buffer-prefix (buffer-name buf)))
              (buffer-list)))

(defun assist--create-chat-buffer (name)
  "Create a new Assist chat buffer with NAME."
  (let ((buffer-name (if (string-prefix-p assist-buffer-prefix name)
                         name
                       (concat assist-buffer-prefix name))))
    (with-current-buffer (get-buffer-create buffer-name)
      (org-mode)
      (assist-minor-mode 1)
      (current-buffer))))

(defun assist--set-status (status)
  "Set the current status to STATUS and update mode line."
  (setq assist--status status)
  (force-mode-line-update))

(defun assist--generate-open-files-context ()
  "Generate a context string describing currently opened file buffers and their projects.
Uses Projectile when available. This context is prepended as a system message so the
LLM can reason about the user's workspace.  Safe for Emacs 27: only uses Projectile APIs."
  (let* ((have-projectile (require 'projectile nil t))
         (file-bufs (seq-filter (lambda (b) (buffer-file-name b)) (buffer-list)))
         (entries
          (cl-loop for b in file-bufs
                   for path = (buffer-file-name b)
                   for project-root = (when have-projectile
                                        (with-current-buffer b
                                          (ignore-errors (projectile-project-root))))
                   for project-name = (cond
                                       (project-root (file-name-nondirectory (directory-file-name project-root)))
                                       (t "(no-project)"))
                   for rel-path = (if (and project-root (string-prefix-p project-root path))
                                      (string-remove-prefix project-root path)
                                    path)
                   collect (format "%s :: %s\n  abs:%s\n  mode:%s modified:%s"
                                   project-name
                                   rel-path
                                   path
                                   (with-current-buffer b (symbol-name major-mode))
                                   (with-current-buffer b (if (buffer-modified-p b) "yes" "no"))))))
    (when entries
      (concat "WORKSPACE CONTEXT (read-only; summarize or reference if helpful)\n"
              "Each entry lists: PROJECT :: RELATIVE-PATH then details.\n"
              (mapconcat #'identity
                         (cl-loop for e in entries
                                  for i from 1
                                  collect (format "%d. %s" i e))
                         "\n")
              "\nEND WORKSPACE CONTEXT"))))

;;; Message parsing

(defun assist--parse-messages ()
  "Parse current buffer into human and AI messages."
  (save-excursion
    (goto-char (point-min))
    (let ((messages '())
          (current-pos (point-min))
          (ai-block-start nil)
          (ai-block-end nil))
      
      ;; Find all AI message blocks
      (while (re-search-forward "^#\\+begin_ai" nil t)
        (setq ai-block-start (line-beginning-position))
        (when (re-search-forward "^#\\+end_ai" nil t)
          (setq ai-block-end (line-end-position))
          
          ;; Add human message before AI block if exists
          (when (< current-pos ai-block-start)
            (let ((human-text (string-trim
                              (buffer-substring-no-properties current-pos ai-block-start))))
              (when (not (string-empty-p human-text))
                (push `((role . "user") (content . ,human-text)) messages))))
          
          ;; Add AI message content (excluding thinking)
          (let ((ai-content (assist--extract-ai-content ai-block-start ai-block-end)))
            (when (not (string-empty-p ai-content))
              (push `((role . "assistant") (content . ,ai-content)) messages)))
          
          (setq current-pos ai-block-end)))
      
      ;; Add final human message if exists
      (when (< current-pos (point-max))
        (let ((human-text (string-trim
                          (buffer-substring-no-properties current-pos (point-max)))))
          (when (not (string-empty-p human-text))
            (push `((role . "user") (content . ,human-text)) messages))))
      
      (nreverse messages))))

(defun assist--extract-ai-content (start end)
  "Extract AI content between START and END, removing thinking sections."
  (save-excursion
    (let ((content (buffer-substring-no-properties start end)))
      ;; Remove the #+begin_ai and #+end_ai lines
      (setq content (replace-regexp-in-string "^#\\+\\(begin\\|end\\)_ai.*$" "" content))
      ;; Remove thinking sections
      (setq content (replace-regexp-in-string "<thinking>\\(\\(.\\|\n\\)*?\\)</thinking>" "" content))
      ;; Clean up multiple newlines and trim
      (setq content (replace-regexp-in-string "\n\n+" "\n\n" content))
      (string-trim content))))

;;; Server communication

(defun assist--send-request (messages success-callback error-callback)
  "Send MESSAGES to Assist server.
SUCCESS-CALLBACK is called on success.
ERROR-CALLBACK is called on error."
  (let* ((payload `((messages . ,(apply 'vector messages))
                    (stream . t)))
         (json-data (json-encode payload))
         (url (concat assist-server-url "/chat/completions"))
         (url-request-method "POST")
         (url-request-extra-headers
          '(("Content-Type" . "application/json; charset=utf-8")))
         (url-request-data (encode-coding-string json-data 'utf-8)))
    (url-retrieve url
                  (lambda (status)
                    (if (plist-get status :error)
                        (funcall error-callback :error-thrown (plist-get status :error))
                      (goto-char (point-min))
                      (re-search-forward "^$")
                      (forward-char 1)
                      (funcall success-callback :data (buffer-substring (point) (point-max)))))
                  nil t)))

;;; Response handling

(defun assist--handle-stream-response (response buffer-name start-point)
  "Handle streaming RESPONSE for BUFFER-NAME starting at START-POINT."
  (when-let ((buffer (get-buffer buffer-name)))
    (with-current-buffer buffer
      (save-excursion
        (goto-char start-point)
        (let ((lines (split-string response "\n"))
	      (data-start "data: "))
          (dolist (line lines)
            (when (string-prefix-p data-start line)
              (if (string= "data: [DONE]" line)
                  ;; Stream is done - finalize response
                  (assist--finalize-response buffer-name (point))
                ;; Process content
                (let* ((json-line (substring line (length data-start)))
                       (data (ignore-errors (json-read-from-string json-line)))
                       (choices (cdr (assoc 'choices data)))
                       (delta (cdr (assoc 'delta (aref choices 0))))
                       (content (cdr (assoc 'content delta))))
                  (when content
                    ;; First content chunk - create AI block
                    (unless (gethash buffer-name assist--active-requests)
                      (assist--ensure-proper-spacing start-point)
                      (insert "#+begin_ai\n")
                      (puthash buffer-name t assist--active-requests)
                      (assist--set-status "assist: processing"))
                    (insert content)))))))))))

(defun assist--ensure-proper-spacing (start-point)
  "Ensure proper spacing before AI block at START-POINT."
  (goto-char start-point)
  (let ((char-before (if (> (point) 1) (char-before) nil))
        (char-before-2 (if (> (point) 2) (char-before (1- (point))) nil)))
    (cond
     ;; No newlines before - add two
     ((not (eq char-before ?\n))
      (insert "\n\n"))
     ;; One newline before - add one more
     ((and (eq char-before ?\n) (not (eq char-before-2 ?\n)))
      (insert "\n"))
     ;; Two or more newlines before - don't add any
     (t nil))))

(defun assist--finalize-response (buffer-name &optional cur-point)
  "Finalize response for BUFFER-NAME, optionally using START-POINT."
  (when-let ((buffer (get-buffer buffer-name)))
    (with-current-buffer buffer
      (save-excursion
        (goto-char (or cur-point
		       (point-max)))
        (insert "\n#+end_ai\n"))
      (remhash buffer-name assist--active-requests)
      (assist--set-status "assist: done")
      (run-with-timer 3 nil (lambda () (assist--set-status "assist"))))))

;;; Core functions

(defun assist-submit ()
  "Submit the conversation in the current buffer to Assist.
Prepends a system message describing currently opened file buffers and their
Projectile projects so the AI can reference the user's active workspace.  The
CUR-POINT at submission time marks where streaming text will be inserted."
  (interactive)
  (unless (derived-mode-p 'org-mode)
    (user-error "Assist submission requires org-mode"))
  
  (let* ((messages (assist--parse-messages))
         (buffer-name (buffer-name))
	 (cur-point (point))
         (context (assist--generate-open-files-context)))
    
    (when (null messages)
      (user-error "No messages to submit"))
    (when context
      (setq messages (cons `((role . "system") (content . ,context)) messages)))
    
    (assist--set-status "assist: submitting")
    (message "assist: submitting")
    
    (assist--send-request
     messages
     (lambda (&rest args)
       (let ((data (plist-get args :data)))
         (assist--handle-stream-response data
					 buffer-name
					 cur-point)))
     (lambda (&rest args)
       (let ((error-thrown (plist-get args :error-thrown)))
         (assist--set-status "assist: error")
         (message "assist: error - %s" error-thrown)
         (remhash buffer-name assist--active-requests))))))

(defun assist-query (query)
  "Submit a one-off QUERY to Assist."
  (interactive "sQuery: ")
  (when (string-empty-p (string-trim query))
    (user-error "Query cannot be empty"))
  
  ;; Get or create target buffer
  (let* ((chat-buffers (mapcar 'buffer-name (assist--get-chat-buffers)))
         (target-buffer-name
          (if chat-buffers
              (completing-read "Chat buffer (or new name): "
                             chat-buffers
                             nil nil
                             (or assist--last-chat-buffer (car chat-buffers)))
            (read-string "New chat name: " "main"))))
    
    ;; Create or switch to buffer
    (let ((buffer (if (member target-buffer-name chat-buffers)
                      (get-buffer target-buffer-name)
                    (assist--create-chat-buffer target-buffer-name))))
      
      (setq assist--last-chat-buffer (buffer-name buffer))
      
      ;; Add query and submit
      (with-current-buffer buffer
        (goto-char (point-max))
        (unless (bolp) (insert "\n"))
        (insert query "\n")
        (assist-submit))
      
      ;; Show buffer when done
      (run-with-timer 1 nil
                      (lambda ()
                        (when (not (gethash (buffer-name buffer) assist--active-requests))
                          (pop-to-buffer buffer)))))))

;;; Minor mode

;;;###autoload
(define-minor-mode assist-minor-mode
  "Minor mode for Assist integration."
  :lighter assist-mode-line-format
  :keymap (let ((map (make-sparse-keymap)))
            (define-key map (kbd "C-c C-a s") #'assist-submit)
            (define-key map (kbd "C-c C-a q") #'assist-query)
            map))

;;;###autoload
(defun assist-enable ()
  "Enable Assist minor mode in current buffer."
  (interactive)
  (assist-minor-mode 1))

;;;###autoload
(defun assist-disable ()
  "Disable Assist minor mode in current buffer."
  (interactive)
  (assist-minor-mode -1))

(provide 'assist)

;;; assist.el ends here
