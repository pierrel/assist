;;; test-assist.el --- Tests for assist.el -*- lexical-binding: t -*-

;;; Commentary:

;; Tests for the Assist.el package.

;;; Code:

(require 'assist)
(require 'ert)

;;; Test utilities

(defmacro with-test-buffer (content &rest body)
  "Execute BODY in a test buffer with CONTENT."
  `(with-temp-buffer
     (org-mode)
     (insert ,content)
     (goto-char (point-min))
     ,@body))

;;; Tests for utility functions

(ert-deftest assist-test-buffer-is-assist-p ()
  "Test buffer recognition."
  (with-temp-buffer
    (rename-buffer "Assist: test")
    (should (assist--buffer-is-assist-p)))
  
  (with-temp-buffer
    (rename-buffer "normal buffer")
    (should-not (assist--buffer-is-assist-p))))

(ert-deftest assist-test-message-parsing-simple ()
  "Test parsing simple human message."
  (with-test-buffer
   "Hello, how are you?"
   (let ((messages (assist--parse-messages)))
     (should (= (length messages) 1))
     (should (string= (cdr (assoc 'role (car messages))) "user"))
     (should (string= (cdr (assoc 'content (car messages))) "Hello, how are you?")))))

(ert-deftest assist-test-message-parsing-with-ai ()
  "Test parsing conversation with AI response."
  (with-test-buffer
   "Hello, how are you?\n\n#+begin_ai\nI'm doing well, thank you!\n#+end_ai\n\nWhat's the weather like?"
   (let ((messages (assist--parse-messages)))
     (should (= (length messages) 3))
     ;; First human message
     (should (string= (cdr (assoc 'role (nth 0 messages))) "user"))
     (should (string= (cdr (assoc 'content (nth 0 messages))) "Hello, how are you?"))
     ;; AI response
     (should (string= (cdr (assoc 'role (nth 1 messages))) "assistant"))
     (should (string= (cdr (assoc 'content (nth 1 messages))) "I'm doing well, thank you!"))
     ;; Second human message
     (should (string= (cdr (assoc 'role (nth 2 messages))) "user"))
     (should (string= (cdr (assoc 'content (nth 2 messages))) "What's the weather like?")))))

(ert-deftest assist-test-message-parsing-with-thinking ()
  "Test parsing AI message with thinking section."
  (with-test-buffer
   "What is 2+2?\n\n#+begin_ai\n<thinking>\nThis is a simple math question. 2+2 equals 4.\n</thinking>\n\nThe answer is 4.\n#+end_ai"
   (let ((messages (assist--parse-messages)))
     (should (= (length messages) 2))
     ;; Human message
     (should (string= (cdr (assoc 'role (nth 0 messages))) "user"))
     (should (string= (cdr (assoc 'content (nth 0 messages))) "What is 2+2?"))
     ;; AI response (thinking removed)
     (should (string= (cdr (assoc 'role (nth 1 messages))) "assistant"))
     (should (string= (cdr (assoc 'content (nth 1 messages))) "The answer is 4.")))))

(ert-deftest assist-test-extract-ai-content ()
  "Test AI content extraction."
  (with-test-buffer
   "#+begin_ai\n<thinking>\nThis is thinking\n</thinking>\n\nActual response\n#+end_ai"
   (let ((content (assist--extract-ai-content (point-min) (point-max))))
     (should (string= content "Actual response")))))

(ert-deftest assist-test-get-chat-buffers ()
  "Test getting chat buffers."
  (let ((buf1 (get-buffer-create "Assist: test1"))
        (buf2 (get-buffer-create "Assist: test2"))
        (buf3 (get-buffer-create "normal buffer")))
    (unwind-protect
        (let ((chat-buffers (assist--get-chat-buffers)))
          (should (member buf1 chat-buffers))
          (should (member buf2 chat-buffers))
          (should-not (member buf3 chat-buffers)))
      ;; Cleanup
      (kill-buffer buf1)
      (kill-buffer buf2)
      (kill-buffer buf3))))

;;; Integration tests

(ert-deftest assist-test-finalize-response ()
  "Test that finalize-response adds end block and cleans up."
  (let ((buffer (get-buffer-create "Test Finalize")))
    (unwind-protect
        (with-current-buffer buffer
          (org-mode)
          (insert "Test content\n#+begin_ai\nSome AI response")
	  (let ((saved-point (point)))
	    (insert "\nsome more information")
            (puthash "Test Finalize" t assist--active-requests)
            (assist--finalize-response "Test Finalize"
				       saved-point)
            (should (string-match-p "#\\+end_ai" (buffer-string)))
            (should-not (gethash "Test Finalize" assist--active-requests))))
      (kill-buffer buffer))))

(ert-deftest assist-test-stream-done-handling ()
  "Test that [DONE] signal triggers finalization."
  (let ((buffer (get-buffer-create "Test Stream Done")))
    (unwind-protect
        (with-current-buffer buffer
          (org-mode)
          (insert "Initial text\n")
          (let ((start-pos (point)))
            ;; Simulate stream with content and DONE (don't pre-set active request)
            (assist--handle-stream-response 
             "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\ndata: [DONE]\n"
             "Test Stream Done" 
             start-pos)
            ;; Check that finalization occurred
            (should (string-match-p "#\\+begin_ai" (buffer-string)))
            (should (string-match-p "#\\+end_ai" (buffer-string)))
            (should (string-match-p "Hello" (buffer-string)))
            (should-not (gethash "Test Stream Done" assist--active-requests))))
      (kill-buffer buffer))))

(ert-deftest assist-test-point-handling-at-end ()
  "Test AI response insertion when point is at end of buffer."
  (let ((buffer (get-buffer-create "Test Point End")))
    (unwind-protect
        (with-current-buffer buffer
          (org-mode)
          (insert "User query")
          (let ((start-pos (point)))
            (assist--handle-stream-response
             "data: {\"choices\":[{\"delta\":{\"content\":\"AI response\"}}]}\ndata: [DONE]\n"
             "Test Point End"
             start-pos)
            ;; Check proper spacing - should have 2 newlines before AI block
            (should (string-match-p "User query\n\n#\\+begin_ai\nAI response\n#\\+end_ai\n" (buffer-string)))))
      (kill-buffer buffer))))

(ert-deftest assist-test-point-handling-with-content-after ()
  "Test AI response insertion when there's content after the point."
  (let ((buffer (get-buffer-create "Test Point Middle")))
    (unwind-protect
        (with-current-buffer buffer
          (org-mode)
          (insert "User query\n\nExisting content below")
          (goto-char (point-min))
          (end-of-line) ; Position after "User query"
          (let ((start-pos (point)))
            (assist--handle-stream-response
             "data: {\"choices\":[{\"delta\":{\"content\":\"AI response\"}}]}\ndata: [DONE]\n"
             "Test Point Middle"
             start-pos)
            ;; Check that spacing is correct and AI response is properly added
            ;; Note: content after the point will remain after the AI response
            (let ((content (buffer-string)))
              (should (string-match-p "User query\n\n#\\+begin_ai\nAI response\n" content))
              (should (string-match-p "#\\+end_ai" content))
              (should (string-match-p "Existing content below" content)))))
      (kill-buffer buffer))))

(ert-deftest assist-test-point-handling-single-newline-present ()
  "Test AI response insertion when there's already one newline."
  (let ((buffer (get-buffer-create "Test Single Newline")))
    (unwind-protect
        (with-current-buffer buffer
          (org-mode)
          (insert "User query\n")
          (let ((start-pos (point)))
            (assist--handle-stream-response
             "data: {\"choices\":[{\"delta\":{\"content\":\"AI response\"}}]}\ndata: [DONE]\n"
             "Test Single Newline"
             start-pos)
            ;; Should add one more newline to make 2 total
            (should (string-match-p "User query\n\n#\\+begin_ai\nAI response\n#\\+end_ai\n" (buffer-string)))))
      (kill-buffer buffer))))

(ert-deftest assist-test-point-handling-double-newline-present ()
  "Test AI response insertion when there are already two newlines."
  (let ((buffer (get-buffer-create "Test Double Newline")))
    (unwind-protect
        (with-current-buffer buffer
          (org-mode)
          (insert "User query\n\n")
          (let ((start-pos (point)))
            (assist--handle-stream-response
             "data: {\"choices\":[{\"delta\":{\"content\":\"AI response\"}}]}\ndata: [DONE]\n"
             "Test Double Newline"
             start-pos)
            ;; Should not add extra newlines
            (should (string-match-p "User query\n\n#\\+begin_ai\nAI response\n#\\+end_ai\n" (buffer-string)))))
      (kill-buffer buffer))))


(ert-deftest assist-test-create-chat-buffer ()
  "Test creating chat buffer."
  (let ((buffer (assist--create-chat-buffer "test")))
    (unwind-protect
        (progn
          (should (bufferp buffer))
          (should (string= (buffer-name buffer) "Assist: test"))
          (with-current-buffer buffer
            (should (derived-mode-p 'org-mode))
            (should assist-minor-mode)))
      (kill-buffer buffer))))

;;; test-assist.el ends here
