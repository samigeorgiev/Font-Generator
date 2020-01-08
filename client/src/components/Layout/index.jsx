import React from "react";

import styles from './index.module.css';

const layout = props => {
    return (
        <>
            <header>

            </header>
            {props.children}
            <footer>
                Copyright
            </footer>
        </>
    );
};

export default layout;